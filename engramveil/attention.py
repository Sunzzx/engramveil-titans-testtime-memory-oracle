"""Short-term memory: causal sliding-window multi-head attention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engramveil.norms import rms_norm


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(np.where(np.isfinite(shifted), shifted, -np.inf))
    denom = np.sum(exp, axis=axis, keepdims=True)
    return np.where(denom > 0, exp / denom, 0.0)


def causal_mask(tq: int, tk: int) -> np.ndarray:
    """Additive mask: 0 on allowed, -inf on future. Shape (tq, tk)."""
    # Allow key positions j where j <= i + (tk - tq) for prefix caches
    offset = tk - tq
    i = np.arange(tq)[:, None]
    j = np.arange(tk)[None, :]
    allowed = j <= (i + offset)
    m = np.zeros((tq, tk), dtype=np.float64)
    m[~allowed] = -np.inf
    return m


def sliding_window_mask(tq: int, tk: int, window: int) -> np.ndarray:
    """Causal + limited lookback of `window` tokens."""
    base = causal_mask(tq, tk)
    offset = tk - tq
    i = np.arange(tq)[:, None]
    j = np.arange(tk)[None, :]
    too_old = j < (i + offset - window + 1)
    base[too_old] = -np.inf
    return base


@dataclass
class SlidingWindowAttention:
    """Fixed-window causal attention — O(T · W · d) compute, O(W · d) cache."""

    dim: int
    n_heads: int = 4
    window: int = 64
    seed: int | None = 0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim={self.dim} not divisible by n_heads={self.n_heads}")
        rng = np.random.default_rng(self.seed)
        d = self.dim
        scale = 1.0 / np.sqrt(d)
        self.W_q = rng.normal(0, scale, size=(d, d)).astype(np.float64)
        self.W_k = rng.normal(0, scale, size=(d, d)).astype(np.float64)
        self.W_v = rng.normal(0, scale, size=(d, d)).astype(np.float64)
        self.W_o = rng.normal(0, scale, size=(d, d)).astype(np.float64)
        self.rms_w = np.ones(d, dtype=np.float64)

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        # (B, T, D) → (B, H, T, Dh)
        b, t, d = x.shape
        h, dh = self.n_heads, self.head_dim
        return x.reshape(b, t, h, dh).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        # (B, H, T, Dh) → (B, T, D)
        b, h, t, dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, t, h * dh)

    def forward(
        self,
        x: np.ndarray,
        *,
        causal: bool = True,
        return_weights: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        x: (B, T, D) or (T, D)
        """
        single = x.ndim == 2
        if single:
            x = x[None, ...]
        b, t, d = x.shape
        if d != self.dim:
            raise ValueError(f"x last dim {d} != dim={self.dim}")

        q = self._split_heads(x @ self.W_q)
        k = self._split_heads(x @ self.W_k)
        v = self._split_heads(x @ self.W_v)

        scale = 1.0 / np.sqrt(float(self.head_dim))
        scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale  # (B,H,T,T)
        if causal:
            mask = sliding_window_mask(t, t, self.window)
            scores = scores + mask[None, None, :, :]
        weights = softmax(scores, axis=-1)
        ctx = np.matmul(weights, v)
        out = self._merge_heads(ctx) @ self.W_o
        out = rms_norm(out, self.rms_w, eps=self.eps)
        if single:
            out = out[0]
            weights = weights[0]
        if return_weights:
            return out, weights
        return out

    def cache_bytes(self, window: int | None = None) -> int:
        """KV cache footprint for one batch item at given window."""
        w = int(window if window is not None else self.window)
        # K and V: each (H, W, Dh) float64
        return 2 * self.n_heads * w * self.head_dim * 8
