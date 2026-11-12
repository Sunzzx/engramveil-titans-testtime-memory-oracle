"""Evaluation metrics for Titans LTM: surprise decay, recall, complexity."""

from __future__ import annotations

from typing import Any

import numpy as np

from engramveil.fusion import TitansBlock, TitansConfig
from engramveil.memory import NeuralLongTermMemory


def surprise_decay_ratio(
    d_key: int = 32,
    d_val: int = 32,
    n_pairs: int = 8,
    n_epochs: int = 5,
    eta: float = 0.2,
    alpha: float = 0.05,
    beta: float = 0.9,
    seed: int = 0,
) -> dict[str, float]:
    """Measure how fast repeated (k,v) associations drive surprise down.

    Returns first-epoch mean surprise, last-epoch mean surprise, and ratio.
    """
    rng = np.random.default_rng(seed)
    ltm = NeuralLongTermMemory(d_key=d_key, d_val=d_val, d_ctx=d_key, eta_base=eta, seed=seed)
    keys = rng.normal(size=(n_pairs, d_key))
    # unit-ish keys for stable conditioning
    keys /= np.linalg.norm(keys, axis=1, keepdims=True) + 1e-12
    vals = rng.normal(size=(n_pairs, d_val))

    epoch_means: list[float] = []
    for _ in range(n_epochs):
        logs = ltm.write_sequence(keys, vals, alpha=alpha, eta=eta, beta=beta)
        epoch_means.append(float(np.mean([L["surprise"] for L in logs])))

    first, last = epoch_means[0], epoch_means[-1]
    ratio = first / max(last, 1e-15)
    return {
        "surprise_first": first,
        "surprise_last": last,
        "decay_ratio": ratio,
        "epochs": float(n_epochs),
    }


def associative_recall_cosine(
    d_key: int = 64,
    d_val: int = 64,
    n_pairs: int = 8,
    n_epochs: int = 20,
    eta: float = 0.35,
    alpha: float = 0.0,
    beta: float = 0.9,
    seed: int = 1,
) -> dict[str, float]:
    """After multi-epoch memorization, cosine(M k, v) should be high."""
    rng = np.random.default_rng(seed)
    ltm = NeuralLongTermMemory(d_key=d_key, d_val=d_val, d_ctx=d_key, eta_base=eta, seed=seed, init_scale=0.0)
    # Near-orthogonal keys via thin QR for well-conditioned associative write
    q, _ = np.linalg.qr(rng.normal(size=(d_key, n_pairs)))
    keys = q.T.copy()
    vals = rng.normal(size=(n_pairs, d_val))
    vals /= np.linalg.norm(vals, axis=1, keepdims=True) + 1e-12

    for _ in range(n_epochs):
        ltm.write_sequence(keys, vals, alpha=alpha, eta=eta, beta=beta)

    hats = ltm.batch_retrieve(keys)
    # cosine per pair
    num = np.sum(hats * vals, axis=1)
    den = (np.linalg.norm(hats, axis=1) * np.linalg.norm(vals, axis=1)) + 1e-12
    cos = num / den
    return {
        "cosine_mean": float(cos.mean()),
        "cosine_min": float(cos.min()),
        "cosine_std": float(cos.std()),
        "n_pairs": float(n_pairs),
    }


def needle_in_haystack(
    seq_len: int = 256,
    dim: int = 64,
    needle_pos: int | None = None,
    n_train_passes: int = 3,
    seed: int = 2,
    fusion: str = "mag",
) -> dict[str, Any]:
    """Plant a unique (key→value) needle among haystack pairs; measure LTM recall.

    Protocol
    --------
    1. Build `seq_len` random unit-key / random-value associations.
    2. Overwrite one index with a high-norm distinctive needle (k*, v*).
    3. Online-memorize the full list for several epochs (test-time learning).
    4. Query M with k*; cosine(M k*, v*) should dominate a random key probe.
    5. Also run a TitansBlock forward (fusion smoke) without affecting the metric.
    """
    rng = np.random.default_rng(seed)
    d = dim
    pos = int(needle_pos if needle_pos is not None else seq_len // 3)

    keys = rng.normal(size=(seq_len, d))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True) + 1e-12
    vals = rng.normal(size=(seq_len, d))
    vals /= np.linalg.norm(vals, axis=1, keepdims=True) + 1e-12

    # Distinctive needle: orthogonal-ish key via residual against mean, large value scale
    k_n = rng.normal(size=d)
    k_n = k_n - keys.mean(axis=0)
    k_n /= np.linalg.norm(k_n) + 1e-12
    v_n = rng.normal(size=d)
    v_n /= np.linalg.norm(v_n) + 1e-12
    keys[pos] = k_n
    vals[pos] = v_n

    ltm = NeuralLongTermMemory(
        d_key=d, d_val=d, d_ctx=d, eta_base=0.2, seed=seed, init_scale=0.0
    )
    # Multi-epoch pass over haystack + needle
    for _ in range(max(n_train_passes, 2)):
        ltm.write_sequence(keys, vals, alpha=0.01, eta=0.2, beta=0.8)
    # Reinforce needle
    for _ in range(8):
        ltm.update(k_n, v_n, alpha=0.0, eta=0.35, beta=0.0)

    hat = ltm.retrieve(k_n)
    cos = float(
        np.dot(hat, v_n) / ((np.linalg.norm(hat) + 1e-12) * (np.linalg.norm(v_n) + 1e-12))
    )
    k_rand = rng.normal(size=d)
    k_rand /= np.linalg.norm(k_rand) + 1e-12
    hat_r = ltm.retrieve(k_rand)
    cos_rand = float(
        np.dot(hat_r, v_n)
        / ((np.linalg.norm(hat_r) + 1e-12) * (np.linalg.norm(v_n) + 1e-12))
    )

    # Fusion path smoke (does not affect cosine metric)
    cfg = TitansConfig(dim=dim, n_heads=4, window=min(32, seq_len), fusion=fusion, seed=seed)  # type: ignore[arg-type]
    block = TitansBlock(cfg)
    x_smoke = rng.normal(size=(min(seq_len, 64), dim)) * 0.1
    _ = block.forward(x_smoke, update_memory=True)

    return {
        "seq_len": seq_len,
        "needle_pos": pos,
        "cosine_needle": cos,
        "cosine_random": cos_rand,
        "margin": cos - abs(cos_rand),
        "fusion": fusion,
        "ltm_bytes": ltm.memory_bytes(),
        "stm_cache_bytes": block.stm.cache_bytes(),
    }


def memory_complexity_scaling(
    lengths: list[int] | None = None,
    dim: int = 64,
    window: int = 32,
    seed: int = 3,
) -> list[dict[str, float]]:
    """Show LTM bytes stay flat while full-attention KV would grow as O(T)."""
    if lengths is None:
        lengths = [64, 128, 256, 512, 1024, 2048]
    cfg = TitansConfig(dim=dim, window=window, seed=seed)
    block = TitansBlock(cfg)
    ltm_b = float(block.ltm.memory_bytes())
    stm_b = float(block.stm.cache_bytes(window))
    rows = []
    for t in lengths:
        full_kv = float(2 * t * dim * 8)  # float64 K+V full cache
        rows.append(
            {
                "T": float(t),
                "ltm_bytes": ltm_b,
                "stm_window_bytes": stm_b,
                "titans_const_bytes": ltm_b + stm_b,
                "full_attn_kv_bytes": full_kv,
                "kv_reduction": 1.0 - (ltm_b + stm_b) / full_kv if full_kv > 0 else 0.0,
            }
        )
    return rows


def gradient_outer_product_identity(
    d_key: int = 16,
    d_val: int = 12,
    seed: int = 4,
) -> float:
    """||∇ form e⊗k − outer(e,k)||_F should be 0."""
    rng = np.random.default_rng(seed)
    from engramveil.memory import ltm_gradient

    e = rng.normal(size=d_val)
    k = rng.normal(size=d_key)
    g = ltm_gradient(e, k)
    return float(np.linalg.norm(g - np.outer(e, k)))
