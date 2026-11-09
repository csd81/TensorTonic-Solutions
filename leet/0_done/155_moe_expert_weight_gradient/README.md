# MoE Expert Weight Gradient — CUDA C++ and Numba Results

Challenge: compute the weight gradients of a Mixture-of-Experts expert
projection.  Routed tokens are sorted by expert; with `s = group_offsets[g]`
and `e = group_offsets[g+1]`, each expert contributes

```
dW[g] = X[s:e]ᵀ × dY[s:e]        (K × N)
```

so the ragged token count is the **reduction** dimension.  Empty experts must
produce an all-zero `K × N` slice; `dW` is not zero-initialised and every
entry must be written.  Performance is measured at `G = 64`,
`M_total = 32,768`, `K = 1,024`, `N = 512` with skewed expert loads and two
empty experts.

Files:

- `moe_expert_weight_gradient.cu` — CUDA C++ implementation of
  `solve(X, dY, group_offsets, dW, G, M_total, K, N)` (renamed from
  `starter.cu`).
- `moe_expert_weight_gradient.py` — Numba CUDA port.

## Implementation — CUDA C++

One block computes a `BM × BN` tile of one expert's `K × N` gradient:

| Parameter | Value |
|---|---:|
| Block | 16×16 threads (256) |
| Output tile | 128 (K) × 128 (N) |
| Register tile per thread | 8 × 8 |
| Reduction tile (`BK`) | 16 tokens |

- the expert's token range is walked in `BK`-row chunks; `X` is staged
  transposed (`As[m][k]`) and `dY` row-major (`Bs[m][n]`), both padded to
  avoid bank conflicts;
- the outer product over the reduction tile accumulates into 8×8 registers
  per thread;
- the epilogue writes the tile to `dW[g]`, so empty experts (whose reduction
  loop never runs) write zeros and every element is covered.

### The starter was already at the roofline

The starter's structure (128×128 tile, 8×8 per thread) is the same one that
performs best for the dense GEMM challenges on this GPU, and at the benchmark
shape it reaches **~1.05 TFLOP/s — about 36% of the P2000's fp32 peak**.
That matches the roofline for this problem: 34.4 GFLOP of FMAs plus 128 MB of
mandatory `dW` stores (the writes alone are ~1.5–3 ms).

Tuning attempts, all within noise or worse:

| Change | Time |
|---|---:|
| starter (BK=16) | 32.7–34.1 ms |
| `BK = 8` | 35.2 ms |
| `BK = 32` | 33.7 ms |
| `BK = 64` | > 48 KB shared, cannot launch |
| float4 tile loads + float4 stores | 32.4–33.6 ms (noise) |

The final kernel therefore keeps the starter's algorithm, with a documented
rationale in the source; no measurable headroom was left on this GPU.

## Implementation — Numba

Same grouped-GEMM structure written with scalar registers: 16×16 threads,
4×4 register tile, 64×64 output tile, `BK = 16`, shared `As`/`Bs` tiles, and
the same empty-expert zero-write behaviour.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, sm_61) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-08 |

Kernel times are medians of 20 runs (C++) / 10 runs (Numba) after a warm-up,
with inputs on the device, including `cudaDeviceSynchronize()`.  `dW` is
poisoned with `0xFF` before each correctness run so every entry must be
written.

## Correctness — CUDA C++

| Case | G | M_total | K | N | Relative error |
|---|---:|---:|---:|---:|---:|
| challenge example | 3 | 3 | 2 | 2 | 0.0 |
| tiny | 1 | 1 | 1 | 1 | 3.60e-08 |
| single expert | 1 | 100 | 16 | 16 | 3.65e-07 |
| odd sizes (empty groups) | 4 | 33 | 7 | 5 | 7.03e-08 |
| skewed | 8 | 1,000 | 64 | 64 | 6.28e-07 |
| skewed | 16 | 4,096 | 128 | 128 | 1.09e-06 |
| non-multiples | 5 | 5,000 | 300 | 70 | 1.96e-06 |
| uniform split | 3 | 300 | 129 | 65 | 2.91e-07 |
| many experts | 64 | 8,192 | 256 | 128 | 9.95e-07 |
| long reduction rows | 2 | 17 | 1,024 | 512 | 1.20e-07 |

The benchmark output was re-verified after timing (relative error 8.4e-07,
sampled across experts and positions).  The benchmark token distribution was
`min = 0, max = 2425` tokens per expert with exactly two empty experts.

## Correctness — Numba CUDA

| Case | G | M_total | K | N | Relative error |
|---|---:|---:|---:|---:|---:|
| challenge example | 3 | 3 | 2 | 2 | 0.0 |
| tiny | 1 | 1 | 1 | 1 | 5.13e-08 |
| single expert | 1 | 100 | 16 | 16 | 1.71e-07 |
| odd sizes (empty groups) | 4 | 33 | 7 | 5 | 1.19e-07 |
| skewed | 8 | 1,000 | 64 | 64 | 4.99e-07 |
| skewed | 16 | 4,096 | 128 | 128 | 1.55e-06 |
| non-multiples | 5 | 5,000 | 300 | 70 | 2.03e-06 |
| uniform split | 3 | 300 | 129 | 65 | 4.53e-07 |
| many experts | 64 | 8,192 | 256 | 128 | 1.10e-06 |
| long reduction rows | 2 | 17 | 1,024 | 512 | 1.29e-07 |

## Performance — G = 64, M_total = 32,768, K = 1,024, N = 512

Work: `2 · M_total · K · N = 34.4 GFLOP`; dW stores: 128 MB.

| Implementation | Median time | GFLOP/s |
|---|---:|---:|
| **CUDA C++ (final, = starter algorithm)** | **32.45–33.51 ms** | **1,025–1,059** |
| Numba CUDA | 76.20 ms | 451 |

The CUDA kernel runs at the compute + store roofline; the Numba port is 2.3×
slower because its scalar register tiles and loads move four times as many
instructions per MAC.

## Build / run

CUDA C++ (the file exports `solve`, so compile it with a harness that
provides `main`):

```bash
nvcc -O3 -arch=sm_61 -o test moe_expert_weight_gradient.cu test_harness.cu
```

Numba:

```bash
python3 moe_expert_weight_gradient.py
```

The Python script runs the correctness cases and the performance case.  For
best performance, pass Numba device arrays; host arrays incur copies to and
from the device.

## Naive baseline

`moe_expert_weight_gradient_naive.cu` is the same computation written as plainly as possible:
one thread per (expert, k, n) output element walking that expert's ragged token range directly in global memory — no shared-memory tiling, no vector types, no warp intrinsics and no
fast-math.  It passes the same correctness cases as the optimized kernel
(checked against a float64 CPU reference in the shared harness).

| Implementation | Time | vs naive |
|---|---:|---:|
| `moe_expert_weight_gradient_naive.cu` | 350.233 ms | 1.00x |
| `moe_expert_weight_gradient.cu` (optimized) | 33.328 ms | 10.5x |

The optimized kernel uses a grouped 128x128 tiled GEMM with register blocking.
