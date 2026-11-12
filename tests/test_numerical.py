"""Numerical stability and edge-case tests."""

from __future__ import annotations

import numpy as np
import pytest

from engramveil.attention import SlidingWindowAttention
from engramveil.fusion import TitansBlock, TitansConfig
from engramveil.memory import NeuralLongTermMemory


def test_ltm_handles_zero_key():
    ltm = NeuralLongTermMemory(d_key=8, d_val=8, seed=0, init_scale=0.1)
    s, e = ltm.surprise(np.zeros(8), np.ones(8))
    assert np.isfinite(s)
    assert e.shape == (8,)
    log = ltm.update(np.zeros(8), np.ones(8), alpha=0.1, eta=0.1, beta=0.5)
    assert np.isfinite(log["surprise"])


def test_ltm_large_eta_remains_finite():
    ltm = NeuralLongTermMemory(d_key=16, d_val=16, seed=1, init_scale=0.01)
    rng = np.random.default_rng(1)
    for _ in range(50):
        k = rng.normal(size=16)
        v = rng.normal(size=16)
        ltm.update(k, v, alpha=0.2, eta=1.0, beta=0.99)
    assert np.all(np.isfinite(ltm.M))
    assert np.all(np.isfinite(ltm.V))


def test_write_sequence_length_mismatch():
    ltm = NeuralLongTermMemory(d_key=4, d_val=4, seed=0)
    with pytest.raises(ValueError):
        ltm.write_sequence(np.zeros((3, 4)), np.zeros((2, 4)))


def test_bad_key_dim_raises():
    ltm = NeuralLongTermMemory(d_key=8, d_val=8, seed=0)
    with pytest.raises(ValueError):
        ltm.retrieve(np.zeros(7))


def test_stm_window_one():
    stm = SlidingWindowAttention(dim=16, n_heads=2, window=1, seed=0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 16))
    y, w = stm.forward(x, causal=True, return_weights=True)
    assert y.shape == x.shape
    # each query attends only to itself
    for t in range(8):
        for h in range(2):
            assert w[h, t, t] == pytest.approx(1.0, abs=1e-10)
            assert w[h, t, :t].sum() == pytest.approx(0.0, abs=1e-10)


def test_titans_batch_dim_rejected():
    cfg = TitansConfig(dim=16, n_heads=2, window=4, seed=0)
    block = TitansBlock(cfg)
    x = np.zeros((2, 5, 16))
    with pytest.raises(ValueError):
        block.forward(x)


def test_persistent_zero_mac():
    cfg = TitansConfig(dim=32, n_heads=4, window=8, fusion="mac", n_persistent=0, seed=0)
    block = TitansBlock(cfg)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 32))
    y = block.forward(x, update_memory=True)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_multi_epoch_orthogonal_keys_near_interpolation():
    """Near-orthogonal keys should allow near-independent storage under enough steps."""
    rng = np.random.default_rng(42)
    d = 64
    n = 4
    # QR for orthogonal keys
    q, _ = np.linalg.qr(rng.normal(size=(d, n)))
    keys = q.T  # (n, d)
    vals = rng.normal(size=(n, d))
    ltm = NeuralLongTermMemory(d_key=d, d_val=d, seed=42, init_scale=0.0, eta_base=0.5)
    for _ in range(40):
        ltm.write_sequence(keys, vals, alpha=0.0, eta=0.5, beta=0.0)
    hats = ltm.batch_retrieve(keys)
    # relative error
    rel = np.linalg.norm(hats - vals, axis=1) / (np.linalg.norm(vals, axis=1) + 1e-12)
    assert float(rel.mean()) < 0.15


def test_gate_history_length_matches_steps():
    ltm = NeuralLongTermMemory(d_key=8, d_val=8, seed=0)
    rng = np.random.default_rng(0)
    n = 17
    for _ in range(n):
        ltm.update(rng.normal(size=8), rng.normal(size=8))
    assert ltm.step_count == n
    assert len(ltm.surprise_history) == n
    assert len(ltm.gate_history) == n
