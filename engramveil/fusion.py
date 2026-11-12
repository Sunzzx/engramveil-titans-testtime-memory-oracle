"""Titans block: LTM + STM fusion via MAG / MAC / MAL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from engramveil.attention import SlidingWindowAttention
from engramveil.memory import NeuralLongTermMemory
from engramveil.norms import rms_norm

FusionMode = Literal["mag", "mac", "mal"]


@dataclass
class TitansConfig:
    dim: int = 64
    d_key: int | None = None
    d_val: int | None = None
    n_heads: int = 4
    window: int = 32
    eta_base: float = 0.15
    n_persistent: int = 4
    fusion: FusionMode = "mag"
    seed: int | None = 0

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.d_key is None:
            self.d_key = self.dim
        if self.d_val is None:
            self.d_val = self.dim
        if self.fusion not in ("mag", "mac", "mal"):
            raise ValueError(f"unknown fusion mode {self.fusion}")
        if self.n_persistent < 0:
            raise ValueError("n_persistent must be >= 0")


class TitansBlock:
    """Full Titans memory branch with persistent prefixes and fusion."""

    def __init__(self, config: TitansConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        d = config.dim
        dk = int(config.d_key)
        dv = int(config.d_val)

        self.ltm = NeuralLongTermMemory(
            d_key=dk,
            d_val=dv,
            d_ctx=d,
            eta_base=config.eta_base,
            seed=config.seed,
        )
        self.stm = SlidingWindowAttention(
            dim=d,
            n_heads=config.n_heads,
            window=config.window,
            seed=None if config.seed is None else config.seed + 1,
        )
        # Project x → (k, v) for LTM writes / reads
        scale = 1.0 / np.sqrt(d)
        self.W_k = rng.normal(0, scale, size=(d, dk)).astype(np.float64)
        self.W_v = rng.normal(0, scale, size=(d, dv)).astype(np.float64)
        self.W_ltm_out = rng.normal(0, scale, size=(dv, d)).astype(np.float64)
        # MAG gate
        self.W_g = rng.normal(0, scale, size=(2 * d, d)).astype(np.float64)
        self.b_g = np.zeros(d, dtype=np.float64)
        # Persistent memory priors (learnable prefix tokens)
        self.persistent = rng.normal(0, 0.02, size=(config.n_persistent, d)).astype(np.float64)
        self.rms_w = np.ones(d, dtype=np.float64)

    def _project_kv(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x @ self.W_k, x @ self.W_v

    def _ltm_read(self, keys: np.ndarray) -> np.ndarray:
        """keys (T, dk) → projected (T, dim)."""
        vals = self.ltm.batch_retrieve(keys)
        return vals @ self.W_ltm_out

    def memorize_sequence(
        self,
        x: np.ndarray,
        *,
        alpha: float | None = None,
        eta: float | None = None,
        beta: float | None = None,
    ) -> list[dict[str, Any]]:
        """Online test-time write of a token sequence into LTM."""
        X = np.asarray(x, dtype=np.float64)
        if X.ndim == 3:
            if X.shape[0] != 1:
                raise ValueError("batch>1 memorize not supported; pass (T,D) or (1,T,D)")
            X = X[0]
        keys, vals = self._project_kv(X)
        return self.ltm.write_sequence(keys, vals, ctxs=X, alpha=alpha, eta=eta, beta=beta)

    def forward(
        self,
        x: np.ndarray,
        *,
        update_memory: bool = True,
        return_aux: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """
        x: (T, D) or (B, T, D) with B=1.

        Pipeline
        --------
        1. Optional persistent prefix concat (MAC path uses it in STM).
        2. STM sliding-window attention.
        3. LTM retrieve (and optional online write).
        4. Fusion (MAG / MAC / MAL).
        """
        single = x.ndim == 2
        X = np.asarray(x, dtype=np.float64)
        if X.ndim == 2:
            X = X[None, ...]
        if X.shape[0] != 1:
            raise ValueError("TitansBlock demo supports batch size 1")
        b, t, d = X.shape
        assert d == self.config.dim

        tokens = X[0]  # (T, D)
        logs: list[dict[str, Any]] = []
        if update_memory:
            logs = self.memorize_sequence(tokens)

        keys, _vals = self._project_kv(tokens)
        y_ltm = self._ltm_read(keys)  # (T, D)

        mode = self.config.fusion
        aux: dict[str, Any] = {"fusion": mode, "ltm_logs": logs}

        if mode == "mal":
            # Memory as Layer: LTM features feed STM
            h = rms_norm(tokens + y_ltm, self.rms_w)
            y_stm = self.stm.forward(h, causal=True)
            y = rms_norm(h + y_stm, self.rms_w)
            aux["y_stm"] = y_stm
            aux["y_ltm"] = y_ltm

        elif mode == "mac":
            # Memory as Context: persistent + LTM summary tokens prepended
            # Use mean LTM readout chunks as extra context prefixes
            n_p = self.config.n_persistent
            if n_p > 0:
                # Mix persistent priors with global LTM summary
                summary = y_ltm.mean(axis=0, keepdims=True)  # (1, D)
                prefixes = self.persistent + 0.1 * summary
                stm_in = np.concatenate([prefixes, tokens], axis=0)
            else:
                stm_in = tokens
                prefixes = np.zeros((0, d))
            y_full = self.stm.forward(stm_in, causal=True)
            y_stm = y_full[n_p:]  # drop prefix outputs
            # Residual fuse with LTM
            y = rms_norm(tokens + y_stm + y_ltm, self.rms_w)
            aux["y_stm"] = y_stm
            aux["y_ltm"] = y_ltm
            aux["n_prefix"] = n_p

        else:  # mag
            y_stm = self.stm.forward(tokens, causal=True)
            gate_in = np.concatenate([y_stm, y_ltm], axis=-1)  # (T, 2D)
            # sigmoid gate per dim
            logits = gate_in @ self.W_g + self.b_g
            g = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
            y = (1.0 - g) * y_stm + g * y_ltm
            y = rms_norm(tokens + y, self.rms_w)
            aux["gate_mean"] = float(g.mean())
            aux["gate_std"] = float(g.std())
            aux["y_stm"] = y_stm
            aux["y_ltm"] = y_ltm

        out = y[None, ...] if not single else y
        if return_aux:
            return out, aux
        return out

    def memory_footprint(self) -> dict[str, int]:
        """Byte footprint: LTM is O(1), STM cache is O(window)."""
        return {
            "ltm_bytes": self.ltm.memory_bytes(),
            "stm_cache_bytes": self.stm.cache_bytes(),
            "persistent_bytes": int(self.persistent.nbytes),
            "total_const_bytes": self.ltm.memory_bytes()
            + self.stm.cache_bytes()
            + int(self.persistent.nbytes),
        }
