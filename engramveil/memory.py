"""Titans Neural Long-Term Memory (LTM).

Core dynamics (Behrouz et al., 2024):

  Prediction:   \\hat{v}_t = M_{t-1} k_t
  Error:        e_t = v_t - \\hat{v}_t
  Surprise:     S_t = (1/2) ||e_t||_2^2
  Gradient:     ∇_M S_t = - e_t ⊗ k_t

  Adaptive gates from context x_t:
    α_t = σ(x_t W_α + b_α)          # forget / weight-decay
    η_t = σ(x_t W_η + b_η) · η_base # learning rate
    β_t = σ(x_t W_β + b_β)          # momentum

  Momentum GD:
    V_t = β_t V_{t-1} + η_t (e_t ⊗ k_t)
    M_t = (1 - α_t) M_{t-1} + V_t

M ∈ R^{d_val × d_key} is a constant-size associative memory — O(1) memory
footprint independent of sequence length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable logistic sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    pos = x >= 0
    out = np.empty_like(x, dtype=np.float64)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def ltm_gradient(error: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Outer-product gradient contribution e ⊗ k (positive form used in update).

    For loss S = 1/2 ||v - M k||^2, ∇_M S = -(v - Mk) k^T = -e k^T.
    The online update *ascends* the negative gradient, i.e. adds η e ⊗ k.
    """
    e = np.asarray(error, dtype=np.float64).reshape(-1)
    k = np.asarray(key, dtype=np.float64).reshape(-1)
    return np.outer(e, k)


@dataclass
class SurpriseGate:
    """Context-conditioned α (forget), η (lr), β (momentum) gates."""

    d_ctx: int
    eta_base: float = 0.1
    seed: int | None = 0
    _W_alpha: np.ndarray = field(init=False, repr=False)
    _b_alpha: np.ndarray = field(init=False, repr=False)
    _W_eta: np.ndarray = field(init=False, repr=False)
    _b_eta: np.ndarray = field(init=False, repr=False)
    _W_beta: np.ndarray = field(init=False, repr=False)
    _b_beta: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(max(self.d_ctx, 1))
        # Bias α low (slow forget), η mid, β high (retain momentum)
        self._W_alpha = rng.normal(0.0, scale, size=(self.d_ctx, 1))
        self._b_alpha = np.array([-1.5], dtype=np.float64)
        self._W_eta = rng.normal(0.0, scale, size=(self.d_ctx, 1))
        self._b_eta = np.array([0.0], dtype=np.float64)
        self._W_beta = rng.normal(0.0, scale, size=(self.d_ctx, 1))
        self._b_beta = np.array([1.5], dtype=np.float64)

    def gates(self, ctx: np.ndarray) -> tuple[float, float, float]:
        """Return (α, η, β) scalars in valid ranges from context vector."""
        x = np.asarray(ctx, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.d_ctx:
            raise ValueError(f"ctx dim {x.shape[0]} != d_ctx={self.d_ctx}")
        alpha = float(sigmoid(x @ self._W_alpha + self._b_alpha)[0])
        eta = float(sigmoid(x @ self._W_eta + self._b_eta)[0]) * self.eta_base
        beta = float(sigmoid(x @ self._W_beta + self._b_beta)[0])
        return alpha, eta, beta


@dataclass
class NeuralLongTermMemory:
    """Constant-size associative LTM updated by surprise-gated momentum GD."""

    d_key: int
    d_val: int
    d_ctx: int | None = None
    eta_base: float = 0.1
    seed: int | None = 0
    init_scale: float = 0.01

    M: np.ndarray = field(init=False, repr=False)
    V: np.ndarray = field(init=False, repr=False)  # momentum buffer
    gate: SurpriseGate = field(init=False, repr=False)
    surprise_history: list[float] = field(default_factory=list, repr=False)
    gate_history: list[tuple[float, float, float]] = field(default_factory=list, repr=False)
    step_count: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.d_key < 1 or self.d_val < 1:
            raise ValueError("d_key and d_val must be >= 1")
        d_ctx = self.d_ctx if self.d_ctx is not None else self.d_key
        self.d_ctx = d_ctx
        rng = np.random.default_rng(self.seed)
        self.M = rng.normal(0.0, self.init_scale, size=(self.d_val, self.d_key)).astype(np.float64)
        self.V = np.zeros_like(self.M)
        self.gate = SurpriseGate(d_ctx=d_ctx, eta_base=self.eta_base, seed=self.seed)

    # ------------------------------------------------------------------ retrieve
    def retrieve(self, key: np.ndarray) -> np.ndarray:
        """\\hat{v} = M k."""
        k = np.asarray(key, dtype=np.float64).reshape(-1)
        if k.shape[0] != self.d_key:
            raise ValueError(f"key dim {k.shape[0]} != d_key={self.d_key}")
        return self.M @ k

    def batch_retrieve(self, keys: np.ndarray) -> np.ndarray:
        """keys: (T, d_key) → (T, d_val)."""
        K = np.asarray(keys, dtype=np.float64)
        if K.ndim != 2 or K.shape[1] != self.d_key:
            raise ValueError(f"keys shape {K.shape} expected (T, {self.d_key})")
        return (self.M @ K.T).T

    # ------------------------------------------------------------------ update
    def surprise(self, key: np.ndarray, value: np.ndarray) -> tuple[float, np.ndarray]:
        """Return (S_t, e_t) without mutating state."""
        k = np.asarray(key, dtype=np.float64).reshape(-1)
        v = np.asarray(value, dtype=np.float64).reshape(-1)
        if v.shape[0] != self.d_val:
            raise ValueError(f"value dim {v.shape[0]} != d_val={self.d_val}")
        hat = self.retrieve(k)
        err = v - hat
        s = 0.5 * float(np.dot(err, err))
        return s, err

    def update(
        self,
        key: np.ndarray,
        value: np.ndarray,
        ctx: np.ndarray | None = None,
        *,
        alpha: float | None = None,
        eta: float | None = None,
        beta: float | None = None,
    ) -> dict[str, Any]:
        """One online LTM step. Returns diagnostics dict."""
        k = np.asarray(key, dtype=np.float64).reshape(-1)
        v = np.asarray(value, dtype=np.float64).reshape(-1)
        if ctx is None:
            ctx = k
        else:
            ctx = np.asarray(ctx, dtype=np.float64).reshape(-1)

        s, err = self.surprise(k, v)
        ga, ge, gb = self.gate.gates(ctx)
        a = float(alpha) if alpha is not None else ga
        e = float(eta) if eta is not None else ge
        b = float(beta) if beta is not None else gb
        # clamp for numerical safety
        a = float(np.clip(a, 0.0, 1.0))
        e = float(max(e, 0.0))
        b = float(np.clip(b, 0.0, 1.0))

        grad = ltm_gradient(err, k)  # e ⊗ k
        # Classical momentum: V stores velocity; apply η·V once per step.
        # (Writing full V each step without factoring η would double-count history.)
        self.V = b * self.V + grad
        self.M = (1.0 - a) * self.M + e * self.V

        self.surprise_history.append(s)
        self.gate_history.append((a, e, b))
        self.step_count += 1
        return {
            "surprise": s,
            "error_norm": float(np.linalg.norm(err)),
            "alpha": a,
            "eta": e,
            "beta": b,
            "M_fro": float(np.linalg.norm(self.M, ord="fro")),
            "V_fro": float(np.linalg.norm(self.V, ord="fro")),
        }

    def write_sequence(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        ctxs: np.ndarray | None = None,
        *,
        alpha: float | None = None,
        eta: float | None = None,
        beta: float | None = None,
    ) -> list[dict[str, Any]]:
        """Sequential test-time writes. keys/values: (T, d)."""
        K = np.asarray(keys, dtype=np.float64)
        V = np.asarray(values, dtype=np.float64)
        if K.shape[0] != V.shape[0]:
            raise ValueError("keys and values length mismatch")
        logs: list[dict[str, Any]] = []
        for t in range(K.shape[0]):
            c = None if ctxs is None else ctxs[t]
            logs.append(self.update(K[t], V[t], c, alpha=alpha, eta=eta, beta=beta))
        return logs

    def reset_momentum(self) -> None:
        self.V = np.zeros_like(self.M)

    def snapshot(self) -> dict[str, np.ndarray]:
        return {"M": self.M.copy(), "V": self.V.copy()}

    def restore(self, snap: dict[str, np.ndarray]) -> None:
        self.M = np.asarray(snap["M"], dtype=np.float64).copy()
        self.V = np.asarray(snap["V"], dtype=np.float64).copy()

    def memory_bytes(self) -> int:
        """Constant memory footprint (M + V), independent of sequence length."""
        return int(self.M.nbytes + self.V.nbytes)
