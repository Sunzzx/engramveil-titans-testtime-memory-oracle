"""Normalization helpers (RMSNorm)."""

from __future__ import annotations

import numpy as np


def rms_norm(x: np.ndarray, weight: np.ndarray | None = None, eps: float = 1e-6) -> np.ndarray:
    """Root-mean-square layer norm over the last axis.

    Parameters
    ----------
    x : (..., D)
    weight : optional (D,) gain
    """
    ms = np.mean(np.square(x), axis=-1, keepdims=True)
    y = x / np.sqrt(ms + eps)
    if weight is not None:
        y = y * weight
    return y
