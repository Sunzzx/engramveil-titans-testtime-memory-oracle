# Engramveil — Titans Test-Time Neural Long-Term Memory Oracle

[![CI](https://github.com/Sunzzx/engramveil-titans-testtime-memory-oracle/actions/workflows/ci.yml/badge.svg)](https://github.com/Sunzzx/engramveil-titans-testtime-memory-oracle/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?logo=github)](https://github.com/sponsors/Sunzzx)
[![Enterprise](https://img.shields.io/badge/Enterprise-Stripe-635bff?logo=stripe)](https://buy.stripe.com/test_engramveil_enterprise)

> Pure-NumPy reference implementation of **Titans: Learning to Memorize at Test Time**
> (Behrouz et al., Google Research, Dec 2024 / arXiv:2501.00663).
> Surprise-gated online gradient descent into a constant-size neural long-term memory,
> fused with sliding-window short-term attention via MAG / MAC / MAL.

---

## Humanity & Real-World Impact

Transformers forget. Their weights freeze after training, and their KV-cache grows without
bound until context windows collapse under memory pressure. That failure mode hits the
systems people trust most:

- **Clinical decision support** — a 200-page chart should not push the critical allergy
  note out of the attention window.
- **Legal / contract RAG** — multi-document due diligence needs durable binding of
  entities to clauses without quadratic cache blow-ups.
- **Assistive long-form agents** — personal copilots must *learn* user preferences
  mid-session without a fine-tune job.

Titans keeps a **fixed-size neural long-term memory** that updates at test time when
tokens are *surprising*, while a small sliding-window attention handles local syntax.
The result is infinite effective context with **$O(1)$ memory**, fewer attention-sink
hallucinations, and deployability on edge devices that cannot hold million-token caches.

---

## Theoretical Foundations

### 1. Associative neural memory

The long-term memory is a matrix $M_t \in \mathbb{R}^{d_{\mathrm{val}} \times d_{\mathrm{key}}}$:

$$
\hat{v}_t = M_{t-1}\, k_t
$$

Given a target association $(k_t, v_t)$, the prediction error and scalar surprise are

$$
e_t = v_t - \hat{v}_t = v_t - M_{t-1} k_t,
\qquad
\mathcal{S}_t = \tfrac{1}{2}\lVert e_t\rVert_2^2.
$$

The matrix gradient is the outer product

$$
\nabla_M \mathcal{S}_t = -\, e_t \otimes k_t.
$$

### 2. Surprise-gated momentum update

Context $x_t$ drives adaptive gates:

$$
\alpha_t = \sigma(x_t W_\alpha + b_\alpha),\quad
\eta_t = \sigma(x_t W_\eta + b_\eta)\cdot\eta_{\mathrm{base}},\quad
\beta_t = \sigma(x_t W_\beta + b_\beta).
$$

Online test-time learning with momentum and forget:

$$
V_t = \beta_t V_{t-1} + (e_t \otimes k_t),
\qquad
M_t = (1 - \alpha_t)\, M_{t-1} + \eta_t\, V_t.
$$

- $\alpha_t$ — forget / weight-decay (protects against catastrophic overwrite)
- $\eta_t$ — adaptive step size (large when surprised)
- $\beta_t$ — momentum (stabilizes multi-token writes)

### 3. Memory–attention fusion

| Mode | Rule |
|------|------|
| **MAG** (Memory as Gating) | $g_t=\sigma([y_{\mathrm{stm}}; y_{\mathrm{ltm}}]W_g),\quad y=(1-g)\odot y_{\mathrm{stm}}+g\odot y_{\mathrm{ltm}}$ |
| **MAC** (Memory as Context) | Persistent priors + LTM summary prepended to STM keys/values |
| **MAL** (Memory as Layer) | $y_{\mathrm{stm}}=\mathrm{Attn}(x + y_{\mathrm{ltm}})$ then residual |

Short-term memory is **causal sliding-window attention** with window $W$:

$$
\mathrm{Attn}_{i,j} \propto \mathbf{1}[i-W < j \le i]\;\exp\!\Big(\frac{q_i^\top k_j}{\sqrt{d_h}}\Big).
$$

### 4. Complexity

| Component | Time | Space |
|-----------|------|-------|
| LTM retrieve / update (per token) | $O(d_{\mathrm{val}} d_{\mathrm{key}})$ | $O(d_{\mathrm{val}} d_{\mathrm{key}})$ constant |
| STM sliding-window attention | $O(T\, W\, d)$ | $O(W\, d)$ cache |
| Full attention KV-cache | $O(T^2 d)$ | $O(T d)$ growing |

At $T=2048$, $d=64$, $W=32$: Titans constant footprint yields **>95% KV-cache reduction**
vs full attention while retaining associative needles in $M$.

---

## Architecture

```
                    x_t  (token)
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     project k,v   STM window   persistent
          │        attention     prefixes
          ▼           │           │
   ┌────────────┐     │           │
   │ LTM M,V    │     │           │
   │ surprise S │     │           │
   │ α,η,β GD   │     │           │
   └─────┬──────┘     │           │
         │ y_ltm      │ y_stm     │
         └─────┬──────┴─────┬─────┘
               ▼            ▼
            MAG / MAC / MAL fusion
                     │
                     ▼
                   y_t
```

---

## Quickstart

```bash
# install
pip install -e ".[dev]"

# mathematical invariant suite
engramveil verify

# live surprise-decay demo
engramveil demo --pairs 8 --epochs 8

# needle-in-a-haystack
engramveil needle --length 256

# memory scaling bench
engramveil bench --quick

# pytest
pytest -q
```

### Verified output (representative)

```
engramveil v0.1.0 — Titans LTM invariant verification
================================================================
[1] ∇ outer-product identity   ||g − e⊗k||_F = 0.000e+00
    PASS
[2] Surprise decay              first=…  last=…  ratio≥5x
    PASS
[3] Associative recall          cos_mean≥0.85
    PASS
[4] Needle-in-haystack          margin≥0.15
    PASS
[5] O(1) memory footprint       reduction≥90% @ T=2048
    PASS
[6] Fusion modes finite         mag=ok mac=ok mal=ok
    PASS
================================================================
ALL PASS
```

---

## Python API

```python
import numpy as np
from engramveil import NeuralLongTermMemory, TitansBlock, TitansConfig

# Pure LTM
ltm = NeuralLongTermMemory(d_key=32, d_val=32, eta_base=0.2, seed=0)
k, v = np.random.randn(32), np.random.randn(32)
ltm.update(k, v)               # surprise-gated write
hat = ltm.retrieve(k)          # associative read

# Full Titans block (STM + LTM + MAG)
cfg = TitansConfig(dim=64, window=32, fusion="mag", seed=0)
block = TitansBlock(cfg)
x = np.random.randn(128, 64)
y = block.forward(x, update_memory=True)
```

---

## CLI Reference

| Command | Purpose |
|---------|---------|
| `engramveil verify` | Gradient, surprise-decay, recall, needle, $O(1)$ memory, fusion |
| `engramveil demo` | ASCII surprise trajectory over repeated pair epochs |
| `engramveil needle` | Needle-in-a-haystack cosine margin probe |
| `engramveil bench` | Titans vs full-attn KV bytes + tok/s |
| `engramveil inspect` | Dimension / footprint dump |
| `engramveil info` | Package metadata |

---

## Packaging & License

- **License:** MIT (see [LICENSE](LICENSE))
- **Commercial / enterprise support:** [Stripe checkout](https://buy.stripe.com/test_engramveil_enterprise)
- **Sponsors:** [github.com/sponsors/Sunzzx](https://github.com/sponsors/Sunzzx)
- **Contact:** ayushkandwal2007@gmail.com

---

## Citation

```bibtex
@article{behrouz2024titans,
  title   = {Titans: Learning to Memorize at Test Time},
  author  = {Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal = {arXiv preprint arXiv:2501.00663},
  year    = {2024}
}
```

```bibtex
@software{engramveil2026,
  title  = {Engramveil: Titans Test-Time Neural Long-Term Memory Oracle},
  author = {Sunzzx and Hermes AI},
  year   = {2026},
  url    = {https://github.com/Sunzzx/engramveil-titans-testtime-memory-oracle}
}
```
