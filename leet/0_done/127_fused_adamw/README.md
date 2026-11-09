# Fused AdamW — CUDA C++ and Numba Results

Challenge: one fused AdamW step, in place:
`p *= (1 − lr·wd)`; `m = β₁m + (1−β₁)g`; `v = β₂v + (1−β₂)g²`;
`p −= lr · (m/(1−β₁ᵗ)) / (√(v/(1−β₂ᵗ)) + ε)`.  `N ≤ 16,777,216`, `t ≥ 1`;
performance is measured at `N = 16,777,216`, `t = 1000`.

Files:

- `fused_adamw.cu` — CUDA C++ implementation of
  `solve(params, grad, m, v, N, lr, beta1, beta2, eps, weight_decay, t)`
  (renamed from `starter.cu`).
- `fused_adamw_naive.cu` — the naive baseline: one thread per element,
  scalar loads/stores.
- `fused_adamw.py` — Numba CUDA port with the same signature.

## Where the time goes at N = 16,777,216

The kernel reads `params`, `grad`, `m`, `v` (4 × 64 MB) and writes `params`,
`m`, `v` (3 × 64 MB) — **448 MB** — so it is memory-bound:

| Variant | Time | GB/s |
|---|---:|---:|
| starter (scalar, one element/thread) | 5.66 ms | 83.0 |
| naive (same shape) | 5.67–5.83 ms | 81–83 |
| **shipped (float4, four elements/thread)** | **5.73 ms** | **82** |
| Numba CUDA | 5.993 ms | 78.4 |

The `sqrtf` and the division are completely hidden behind the 448 MB of
traffic; the two forms tie.  The shipped version is the float4 one — fewer
instructions for the same bytes — with the `N % 4` tail handled in-kernel and
a scalar kernel for misaligned pointers.

## Implementation — CUDA C++

Four elements per thread via `float4` loads/stores; the bias corrections are
computed once on the host (`1 − βᵗ`, reciprocal), so the device does
`m *= ibc1`, `v *= ibc2` instead of two per-element divisions.  The
decoupled weight decay is applied to the pre-update value, in the order the
challenge specifies.  IEEE division and `sqrtf` match the starter's math.

### Naive baseline

`fused_adamw_naive.cu` is the plainest correct form — the starter's scalar
shape: one thread per element, four scalar loads, three scalar stores, and
the per-element bias-correction divisions.  It measures **5.67–5.83 ms**, the
same as the shipped kernel within noise.

## Implementation — Numba

Scalar grid-stride kernel with `math.sqrt` (float32) and IEEE division.
It reaches **5.993 ms (78.4 GB/s)**, 4.6% behind the C++ kernel — the usual
codegen gap at this size.  Kernels use `cache=True`.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, 6 SMs) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Times are medians after a warm-up, including `cudaDeviceSynchronize()`; the
C++ variants were compiled against the same harness and run interleaved.

## Correctness — CUDA C++

Checked against a float64 CPU reference for all three updated arrays
(`params`, `m`, `v`) with per-element tolerance `1e-5 + 1e-4·|ref|`:

| Case | Result |
|---|---|
| challenge example 1 (N=2, t=1, wd=0) | exact |
| challenge example 2 (N=1, t=1, wd=0.01) | exact |
| N=1, t=1; N=1000, t=1; N=100,000, t=1000; N=100,000, t=100,000 | max_abs ≤ 1e-06 |
| benchmark shape (N=16,777,216, t=1000) | max_abs ≤ 1e-05 |

All 7 harness cases pass.

## Correctness — Numba CUDA

All 6 cases pass with the same tolerance.

## Performance — N = 16,777,216, t = 1000

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| starter (scalar) | 5.66 ms | 83.0 | 1.00× |
| `fused_adamw_naive.cu` (same shape) | 5.67–5.83 ms | 81–83 | 1.00× |
| **`fused_adamw.cu`** (float4) | **5.73 ms** | **82** | 1.0× |
| Numba CUDA | 5.993 ms | 78.4 | 1.05× slower |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test fused_adamw.cu test_harness.cu
```

Numba:

```bash
python3 fused_adamw.py
```

The script runs the correctness cases and the performance case.
