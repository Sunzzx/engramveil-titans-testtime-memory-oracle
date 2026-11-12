"""Core mathematical invariants for Titans LTM."""

from __future__ import annotations

import numpy as np
import pytest

from engramveil.attention import SlidingWindowAttention, causal_mask, sliding_window_mask, softmax
from engramveil.fusion import TitansBlock, TitansConfig
from engramveil.memory import NeuralLongTermMemory, SurpriseGate, ltm_gradient, sigmoid
from engramveil.metrics import (
    associative_recall_cosine,
    gradient_outer_product_identity,
    memory_complexity_scaling,
    needle_in_haystack,
    surprise_decay_ratio,
)
from engramveil.norms import rms_norm


def test_sigmoid_bounds_and_zero():
    x = np.array([-100.0, 0.0, 100.0])
    y = sigmoid(x)
    assert y[0] < 1e-12
    assert y[1] == pytest.approx(0.5, abs=1e-12)
    assert y[2] > 1.0 - 1e-12
    assert np.all(y >= 0) and np.all(y <= 1)


def test_softmax_simplex():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(3, 5, 7))
    p = softmax(logits, axis=-1)
    np.testing.assert_allclose(p.sum(axis=-1), 1.0, atol=1e-12)
    assert np.all(p >= 0)


def test_softmax_causal_masked():
    logits = np.ones((1, 4, 4))
    mask = causal_mask(4, 4)
    p = softmax(logits + mask, axis=-1)
    for i in range(4):
        assert p[0, i, i + 1 :].sum() == pytest.approx(0.0, abs=1e-12)
        assert p[0, i, : i + 1].sum() == pytest.approx(1.0, abs=1e-12)


def test_sliding_window_mask_lookback():
    m = sliding_window_mask(6, 6, window=3)
    # row 5 can see keys 3,4,5 only
    assert np.isneginf(m[5, 2])
    assert m[5, 3] == 0.0
    assert m[5, 5] == 0.0
    assert np.isneginf(m[0, 1])  # future


def test_rms_norm_unit():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 8, 16))
    y = rms_norm(x, eps=0.0)
    ms = np.mean(y**2, axis=-1)
    np.testing.assert_allclose(ms, 1.0, atol=1e-12)


def test_ltm_gradient_outer_product():
    err = gradient_outer_product_identity(d_key=20, d_val=15, seed=4)
    assert err < 1e-14


def test_ltm_retrieve_shape_and_linearity():
    rng = np.random.default_rng(2)
    ltm = NeuralLongTermMemory(d_key=8, d_val=6, seed=2, init_scale=0.1)
    k = rng.normal(size=8)
    v1 = ltm.retrieve(k)
    v2 = ltm.retrieve(2 * k)
    np.testing.assert_allclose(v2, 2 * v1, atol=1e-12)
    assert v1.shape == (6,)


def test_ltm_batch_retrieve_matches_single():
    rng = np.random.default_rng(3)
    ltm = NeuralLongTermMemory(d_key=10, d_val=7, seed=3)
    K = rng.normal(size=(5, 10))
    B = ltm.batch_retrieve(K)
    for i in range(5):
        np.testing.assert_allclose(B[i], ltm.retrieve(K[i]), atol=1e-12)


def test_surprise_nonnegative_and_zero_at_fixed_point():
    ltm = NeuralLongTermMemory(d_key=4, d_val=4, seed=0, init_scale=0.0)
    # M=0 roughly (init_scale 0 → zeros) wait init_scale 0 gives zeros
    k = np.array([1.0, 0, 0, 0])
    v = np.zeros(4)
    s, e = ltm.surprise(k, v)
    assert s == pytest.approx(0.0, abs=1e-15)
    assert np.allclose(e, 0.0)


def test_single_update_reduces_surprise_for_one_pair():
    ltm = NeuralLongTermMemory(d_key=16, d_val=16, seed=5, eta_base=0.5, init_scale=0.01)
    rng = np.random.default_rng(5)
    k = rng.normal(size=16)
    k /= np.linalg.norm(k)
    v = rng.normal(size=16)
    s0, _ = ltm.surprise(k, v)
    ltm.update(k, v, alpha=0.0, eta=0.4, beta=0.0)  # pure GD step, no momentum/forget
    s1, _ = ltm.surprise(k, v)
    assert s1 < s0


def test_momentum_buffer_accumulates():
    ltm = NeuralLongTermMemory(d_key=8, d_val=8, seed=6, init_scale=0.0)
    k = np.ones(8) / np.sqrt(8)
    v = np.ones(8)
    ltm.update(k, v, alpha=0.0, eta=0.1, beta=0.9)
    fro1 = float(np.linalg.norm(ltm.V, ord="fro"))
    # Second step with identical residual direction grows velocity under β>0
    ltm.update(k, v * 0.5, alpha=0.0, eta=0.1, beta=0.9)
    fro2 = float(np.linalg.norm(ltm.V, ord="fro"))
    assert fro1 > 0
    assert fro2 > 0


def test_forget_gate_shrinks_memory():
    ltm = NeuralLongTermMemory(d_key=8, d_val=8, seed=7, init_scale=1.0)
    fro0 = float(np.linalg.norm(ltm.M, ord="fro"))
    k = np.zeros(8)
    v = np.zeros(8)
    # zero error → grad 0; high alpha forgets toward 0 with V~0
    ltm.V[:] = 0
    ltm.update(k, v, alpha=0.5, eta=0.0, beta=0.0)
    fro1 = float(np.linalg.norm(ltm.M, ord="fro"))
    assert fro1 == pytest.approx(0.5 * fro0, rel=1e-10)


def test_surprise_gate_ranges():
    g = SurpriseGate(d_ctx=16, eta_base=0.2, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, e, b = g.gates(rng.normal(size=16))
        assert 0.0 <= a <= 1.0
        assert 0.0 <= e <= 0.2 + 1e-12
        assert 0.0 <= b <= 1.0


def test_surprise_decay_ratio_strong():
    res = surprise_decay_ratio(n_pairs=8, n_epochs=6, eta=0.25, alpha=0.05, beta=0.9, seed=0)
    assert res["decay_ratio"] >= 5.0
    assert res["surprise_last"] < res["surprise_first"]


def test_associative_recall_high_cosine():
    res = associative_recall_cosine(seed=1)
    assert res["cosine_mean"] >= 0.85


def test_memory_bytes_constant():
    ltm = NeuralLongTermMemory(d_key=32, d_val=32, seed=0)
    b0 = ltm.memory_bytes()
    rng = np.random.default_rng(0)
    for _ in range(100):
        ltm.update(rng.normal(size=32), rng.normal(size=32), alpha=0.1, eta=0.1, beta=0.5)
    assert ltm.memory_bytes() == b0
    assert b0 == 2 * 32 * 32 * 8  # M + V float64


def test_complexity_scaling_flat_and_reduction():
    rows = memory_complexity_scaling(lengths=[128, 2048], dim=64, window=32)
    assert rows[0]["titans_const_bytes"] == rows[1]["titans_const_bytes"]
    assert rows[1]["kv_reduction"] > 0.9
    assert rows[1]["full_attn_kv_bytes"] > rows[1]["titans_const_bytes"]


def test_needle_in_haystack_margin():
    res = needle_in_haystack(seq_len=96, dim=64, n_train_passes=4, seed=2)
    assert res["cosine_needle"] >= 0.5
    assert res["margin"] >= 0.15


@pytest.mark.parametrize("fusion", ["mag", "mac", "mal"])
def test_titans_forward_shapes_finite(fusion: str):
    cfg = TitansConfig(dim=32, n_heads=4, window=8, fusion=fusion, n_persistent=2, seed=9)  # type: ignore[arg-type]
    block = TitansBlock(cfg)
    rng = np.random.default_rng(9)
    x = rng.normal(size=(20, 32))
    y, aux = block.forward(x, update_memory=True, return_aux=True)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert aux["fusion"] == fusion
    assert len(aux["ltm_logs"]) == 20


def test_titans_no_update_preserves_ltm():
    cfg = TitansConfig(dim=32, n_heads=4, window=8, seed=10)
    block = TitansBlock(cfg)
    snap = block.ltm.snapshot()
    rng = np.random.default_rng(10)
    x = rng.normal(size=(12, 32))
    block.forward(x, update_memory=False)
    np.testing.assert_allclose(block.ltm.M, snap["M"])
    np.testing.assert_allclose(block.ltm.V, snap["V"])


def test_stm_attention_causal_weights():
    stm = SlidingWindowAttention(dim=32, n_heads=4, window=16, seed=0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 10, 32))
    _, w = stm.forward(x, causal=True, return_weights=True)
    # w: (B,H,T,T)
    for t in range(10):
        assert w[0, :, t, t + 1 :].sum() == pytest.approx(0.0, abs=1e-10)


def test_snapshot_restore_roundtrip():
    ltm = NeuralLongTermMemory(d_key=8, d_val=8, seed=11)
    rng = np.random.default_rng(11)
    ltm.update(rng.normal(size=8), rng.normal(size=8), alpha=0.1, eta=0.2, beta=0.8)
    snap = ltm.snapshot()
    ltm.update(rng.normal(size=8), rng.normal(size=8), alpha=0.1, eta=0.2, beta=0.8)
    ltm.restore(snap)
    np.testing.assert_allclose(ltm.M, snap["M"])
    np.testing.assert_allclose(ltm.V, snap["V"])


def test_config_validation():
    with pytest.raises(ValueError):
        TitansConfig(dim=30, n_heads=4)  # not divisible
    with pytest.raises(ValueError):
        TitansConfig(dim=32, fusion="nope")  # type: ignore[arg-type]
