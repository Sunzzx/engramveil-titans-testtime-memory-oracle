"""Interactive CLI for Engramveil Titans LTM."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from engramveil import __version__
from engramveil.fusion import TitansBlock, TitansConfig
from engramveil.memory import NeuralLongTermMemory
from engramveil.metrics import (
    associative_recall_cosine,
    memory_complexity_scaling,
    needle_in_haystack,
    surprise_decay_ratio,
)


def _bar(val: float, width: int = 40) -> str:
    v = max(0.0, min(1.0, val))
    n = int(round(v * width))
    return "█" * n + "·" * (width - n)


def _spark(vals: list[float], width: int = 48) -> str:
    if not vals:
        return ""
    blocks = " ·░▒▓█"
    arr = np.asarray(vals, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    span = hi - lo if hi > lo else 1.0
    idx = np.linspace(0, len(arr) - 1, min(width, len(arr))).astype(int)
    chars = []
    for i in idx:
        norm = (arr[i] - lo) / span
        bi = int(round(norm * (len(blocks) - 1)))
        chars.append(blocks[bi])
    return "".join(chars)


def cmd_verify(_: argparse.Namespace) -> int:
    print(f"engramveil v{__version__} — Titans LTM invariant verification")
    print("=" * 64)
    ok = True

    # 1. Outer-product gradient identity
    from engramveil.metrics import gradient_outer_product_identity

    gerr = gradient_outer_product_identity()
    print(f"[1] ∇ outer-product identity   ||g − e⊗k||_F = {gerr:.3e}")
    if gerr > 1e-12:
        print("    FAIL")
        ok = False
    else:
        print("    PASS")

    # 2. Surprise decay on repeated pairs
    dec = surprise_decay_ratio(n_pairs=8, n_epochs=6, eta=0.25, alpha=0.05, beta=0.9, seed=0)
    print(
        f"[2] Surprise decay              first={dec['surprise_first']:.4f}  "
        f"last={dec['surprise_last']:.4f}  ratio={dec['decay_ratio']:.1f}x"
    )
    if dec["decay_ratio"] < 5.0 or dec["surprise_last"] >= dec["surprise_first"]:
        print("    FAIL (expected ≥5x decay)")
        ok = False
    else:
        print("    PASS")

    # 3. Associative recall cosine
    rec = associative_recall_cosine(seed=1)
    print(
        f"[3] Associative recall          cos_mean={rec['cosine_mean']:.4f}  "
        f"cos_min={rec['cosine_min']:.4f}"
    )
    if rec["cosine_mean"] < 0.85:
        print("    FAIL (expected cos_mean ≥ 0.85)")
        ok = False
    else:
        print("    PASS")

    # 4. Needle-in-haystack margin
    needle = needle_in_haystack(seq_len=128, dim=64, n_train_passes=4, seed=2)
    print(
        f"[4] Needle-in-haystack          cos_n={needle['cosine_needle']:.4f}  "
        f"cos_r={needle['cosine_random']:.4f}  margin={needle['margin']:.4f}"
    )
    if needle["cosine_needle"] < 0.5 or needle["margin"] < 0.15:
        print("    FAIL (expected strong needle preference)")
        ok = False
    else:
        print("    PASS")

    # 5. Constant memory footprint
    rows = memory_complexity_scaling(lengths=[256, 2048], dim=64, window=32)
    r0, r1 = rows[0], rows[1]
    flat = abs(r0["titans_const_bytes"] - r1["titans_const_bytes"]) < 1e-9
    print(
        f"[5] O(1) memory footprint       T=256→{r0['titans_const_bytes']:.0f}B  "
        f"T=2048→{r1['titans_const_bytes']:.0f}B  "
        f"full_attn@2048={r1['full_attn_kv_bytes']:.0f}B  "
        f"reduction={r1['kv_reduction']*100:.1f}%"
    )
    if not flat or r1["kv_reduction"] < 0.9:
        print("    FAIL")
        ok = False
    else:
        print("    PASS")

    # 6. Fusion modes produce finite outputs
    rng = np.random.default_rng(7)
    x = rng.normal(size=(24, 64)) * 0.5
    print("[6] Fusion modes finite         ", end="")
    mode_ok = True
    for mode in ("mag", "mac", "mal"):
        cfg = TitansConfig(dim=64, n_heads=4, window=16, fusion=mode, seed=7)  # type: ignore[arg-type]
        block = TitansBlock(cfg)
        y = block.forward(x, update_memory=True)
        if not np.all(np.isfinite(y)) or y.shape != x.shape:
            mode_ok = False
            print(f"{mode}=BAD ", end="")
        else:
            print(f"{mode}=ok ", end="")
    print()
    if not mode_ok:
        print("    FAIL")
        ok = False
    else:
        print("    PASS")

    print("=" * 64)
    if ok:
        print("ALL PASS")
        return 0
    print("SOME FAILURES")
    return 1


def cmd_demo(args: argparse.Namespace) -> int:
    print(f"engramveil v{__version__} — live surprise-gated memorization demo")
    print("=" * 64)
    rng = np.random.default_rng(args.seed)
    d_key = args.dim
    d_val = args.dim
    n_pairs = args.pairs
    ltm = NeuralLongTermMemory(
        d_key=d_key, d_val=d_val, d_ctx=d_key, eta_base=args.eta, seed=args.seed
    )
    keys = rng.normal(size=(n_pairs, d_key))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True) + 1e-12
    vals = rng.normal(size=(n_pairs, d_val))

    print(f"pairs={n_pairs}  dim={d_key}  epochs={args.epochs}  η={args.eta}  α={args.alpha}  β={args.beta}")
    print()
    epoch_surp: list[float] = []
    for ep in range(args.epochs):
        logs = ltm.write_sequence(keys, vals, alpha=args.alpha, eta=args.eta, beta=args.beta)
        s = float(np.mean([L["surprise"] for L in logs]))
        epoch_surp.append(s)
        # live bar relative to first epoch
        rel = s / max(epoch_surp[0], 1e-12)
        print(f"  epoch {ep+1:02d}  surprise={s:10.6f}  {_bar(1.0 - min(rel, 1.0))}  rel={rel:.3f}")
    print()
    print(f"surprise trajectory: {_spark(epoch_surp)}")
    print(f"decay ratio: {epoch_surp[0] / max(epoch_surp[-1], 1e-15):.1f}x")

    hats = ltm.batch_retrieve(keys)
    cos = []
    for i in range(n_pairs):
        num = float(np.dot(hats[i], vals[i]))
        den = (np.linalg.norm(hats[i]) + 1e-12) * (np.linalg.norm(vals[i]) + 1e-12)
        cos.append(num / den)
    print(f"recall cosine mean={np.mean(cos):.4f}  min={np.min(cos):.4f}")
    print(f"LTM footprint: {ltm.memory_bytes()} bytes (constant)")
    return 0


def cmd_needle(args: argparse.Namespace) -> int:
    print(f"engramveil v{__version__} — needle-in-a-haystack LTM probe")
    print("=" * 64)
    t0 = time.perf_counter()
    res = needle_in_haystack(
        seq_len=args.length,
        dim=args.dim,
        needle_pos=args.pos,
        n_train_passes=args.passes,
        seed=args.seed,
        fusion=args.fusion,
    )
    dt = time.perf_counter() - t0
    print(f"seq_len={res['seq_len']}  needle_pos={res['needle_pos']}  fusion={res['fusion']}")
    print(f"cosine(needle key → value)  = {res['cosine_needle']:.4f}  {_bar((res['cosine_needle']+1)/2)}")
    print(f"cosine(random key → value)  = {res['cosine_random']:.4f}")
    print(f"margin                      = {res['margin']:.4f}")
    print(f"LTM bytes={res['ltm_bytes']}  STM window cache={res['stm_cache_bytes']}  ({dt*1000:.1f} ms)")
    return 0 if res["cosine_needle"] >= 0.5 and res["margin"] >= 0.1 else 1


def cmd_bench(args: argparse.Namespace) -> int:
    print(f"engramveil v{__version__} — memory complexity bench")
    print("=" * 64)
    lengths = [64, 128, 256, 512, 1024, 2048, 4096]
    if args.quick:
        lengths = [128, 512, 2048]
    rows = memory_complexity_scaling(lengths=lengths, dim=args.dim, window=args.window)
    print(f"{'T':>8}  {'Titans B':>12}  {'FullAttn KV':>14}  {'Reduction':>10}")
    print("-" * 50)
    for r in rows:
        print(
            f"{int(r['T']):>8}  {r['titans_const_bytes']:>12.0f}  "
            f"{r['full_attn_kv_bytes']:>14.0f}  {r['kv_reduction']*100:>9.2f}%"
        )

    # throughput: sequential LTM writes
    rng = np.random.default_rng(0)
    ltm = NeuralLongTermMemory(d_key=args.dim, d_val=args.dim, eta_base=0.1, seed=0)
    n = 2000 if not args.quick else 500
    K = rng.normal(size=(n, args.dim))
    V = rng.normal(size=(n, args.dim))
    t0 = time.perf_counter()
    ltm.write_sequence(K, V, alpha=0.05, eta=0.1, beta=0.9)
    dt = time.perf_counter() - t0
    print()
    print(f"LTM write throughput: {n/dt:.0f} tok/s  ({n} tokens in {dt*1000:.1f} ms)")

    # forward pass
    cfg = TitansConfig(dim=args.dim, window=args.window, seed=0)
    block = TitansBlock(cfg)
    x = rng.normal(size=(min(256, lengths[-1]), args.dim))
    t0 = time.perf_counter()
    for _ in range(5):
        block.forward(x, update_memory=True)
    dt = (time.perf_counter() - t0) / 5
    print(f"TitansBlock forward:  {x.shape[0]/dt:.0f} tok/s  (T={x.shape[0]}, dim={args.dim})")
    return 0


def cmd_inspect(_: argparse.Namespace) -> int:
    print(f"engramveil v{__version__} — architecture inspect")
    print("=" * 64)
    cfg = TitansConfig()
    block = TitansBlock(cfg)
    fp = block.memory_footprint()
    print("TitansConfig defaults:")
    print(f"  dim={cfg.dim}  d_key={cfg.d_key}  d_val={cfg.d_val}")
    print(f"  n_heads={cfg.n_heads}  window={cfg.window}  fusion={cfg.fusion}")
    print(f"  n_persistent={cfg.n_persistent}  eta_base={cfg.eta_base}")
    print()
    print("Memory footprint (constant w.r.t. sequence length):")
    for k, v in fp.items():
        print(f"  {k}: {v} bytes ({v/1024:.2f} KiB)")
    print()
    print("Update equations:")
    print("  hat_v = M k")
    print("  e = v - hat_v ;  S = 1/2 ||e||^2")
    print("  V ← β V + (e ⊗ k)")
    print("  M ← (1-α) M + η V")
    print("  MAG: y = (1-g)⊙y_stm + g⊙y_ltm")
    return 0


def cmd_info(_: argparse.Namespace) -> int:
    print(
        f"""engramveil-titans-testtime-memory-oracle v{__version__}
Pillar: Frontier AI / ML — Titans test-time neural long-term memory
Paper: Behrouz et al., Google Research (Dec 2024) arXiv:2501.00663
GitHub: https://github.com/Sunzzx/engramveil-titans-testtime-memory-oracle
"""
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="engramveil",
        description="Titans test-time neural long-term memory oracle (Engramveil)",
    )
    p.add_argument("--version", action="version", version=f"engramveil {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("verify", help="Run mathematical invariant suite")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("demo", help="Live surprise-decay memorization demo")
    sp.add_argument("--dim", type=int, default=32)
    sp.add_argument("--pairs", type=int, default=8)
    sp.add_argument("--epochs", type=int, default=8)
    sp.add_argument("--eta", type=float, default=0.25)
    sp.add_argument("--alpha", type=float, default=0.05)
    sp.add_argument("--beta", type=float, default=0.9)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("needle", help="Needle-in-a-haystack LTM probe")
    sp.add_argument("--length", type=int, default=256)
    sp.add_argument("--dim", type=int, default=64)
    sp.add_argument("--pos", type=int, default=None)
    sp.add_argument("--passes", type=int, default=4)
    sp.add_argument("--fusion", choices=["mag", "mac", "mal"], default="mag")
    sp.add_argument("--seed", type=int, default=2)
    sp.set_defaults(func=cmd_needle)

    sp = sub.add_parser("bench", help="Memory scaling + throughput benchmarks")
    sp.add_argument("--dim", type=int, default=64)
    sp.add_argument("--window", type=int, default=32)
    sp.add_argument("--quick", action="store_true")
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("inspect", help="Print architecture dimensions")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("info", help="Project metadata")
    sp.set_defaults(func=cmd_info)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
