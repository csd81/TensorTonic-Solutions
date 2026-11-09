# Weight Dequantization — CUDA C++ and Numba Results

Challenge: `Y[i][j] = X[i][j] × S[i/T][j/T]`, where `X` is `M×N` and the
scale matrix `S` is `ceil(M/T) × ceil(N/T)`.  `M, N ≤ 8192`;
`T ∈ {16, 32, 64, 128}`; performance is measured at `8192 × 8192`, `T = 128`.

Files:

- `weight_dequantization.cu` — CUDA C++ implementation of
  `solve(X, S, Y, M, N, TILE_SIZE)` (renamed from `starter.cu`).
- `weight_dequantization_naive.cu` — the naive baseline: one thread per
  element with a scalar load/store and two integer divisions.
- `weight_dequantization.py` — Numba CUDA port with the same signature.

## Where the time goes at 8192 × 8192, T = 128

The kernel reads 268 MB and writes 268 MB — 537 MB — while `S` is a 64×64
table (16 KB) that stays in cache.  A float4 copy of the same traffic
measures **6.55–6.72 ms (80–82 GB/s)**, and the implementations line up
against it:

| Variant | Time | GB/s |
|---|---:|---:|
| starter (scalar, 16×16 blocks) | 7.35 ms | 73.1 |
| naive (same shape) | 7.29–7.33 ms | 73.3–73.6 |
| **shipped (float4, 4 cells/thread, 64×4 blocks)** | **6.51–6.96 ms** | **77–82** |
| float4 copy floor | 6.55–6.72 ms | 80–82 |

The starter pays for one scalar load, one scalar store and two integer
divisions per element; the shipped kernel does one float4 load/store and one
`S` lookup per four cells (all four always share a tile because `T` and the
4-cell groups are multiples of 4).

## Implementation — CUDA C++

Four adjacent columns per thread, 64×4 blocks: one `float4` load, one
`__ldg` of `S[(i/T)*S_cols + j/T]`, one `float4` store.  A ragged tail group
(group crossing the last column) and the scalar fallback handle
`T % 4 != 0`, `N % 4 != 0`, or misaligned pointers.

An alternative that replaces the divisions with shifts (valid because every
allowed `T` is a power of two) was measured at 6.57–6.60 ms — the same
within noise, since the vector kernel hides one division per four elements.
The simpler division form is kept in C++ (it also works for any `T`).

### Naive baseline

`weight_dequantization_naive.cu` is the plainest correct form — the starter's
scalar shape: one thread per element, 16×16 blocks, scalar load/store and two
integer divisions.  It measures **7.3 ms**, ~11% slower than the shipped
kernel.

## Implementation — Numba

Scalar one-element-per-thread with 32×8 blocks.  Two Numba-specific costs
had to be removed to reach parity:

1. `blockIdx`/`threadIdx` arithmetic promotes to **int64**, and the two
   per-element 64-bit divisions took the port to 14.1 ms.  Forcing the
   indices to int32 brought it to 12.5 ms.
2. Numba's remaining 32-bit integer division is still far more expensive than
   nvcc's.  Every allowed tile size is a power of two, so the port shifts
   instead of dividing (`i >> log2(T)`), with a division kernel kept as a
   fallback for other `T`.  That reaches **7.06 ms (76.1 GB/s)** — the same
   as the C++ scalar form.

Both kernels use `cache=True`.

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

Checked against a float64 CPU reference with relative tolerance 1e-4:

| Case | Result |
|---|---|
| challenge example (4×4, T=2) | exact |
| 1×1, 16×16, 17×33, 100×37, 256×128, 1000×1000 for each T ∈ {16,32,64,128} | max_abs ≤ 2e-06 |
| misaligned `X+1`, `Y+1` (1000×1000, T=16) | max_abs 1.9e-06 |
| benchmark shape (8192×8192, T=128) | max_abs ≤ 2e-06 |

All 27 harness cases pass.

## Correctness — Numba CUDA

All 21 cases pass (example, five shapes × four tile sizes).

## Performance — 8192 × 8192, T = 128

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| starter (scalar, 16×16) | 7.35 ms | 73.1 | 1.07–1.11× slower |
| `weight_dequantization_naive.cu` (same shape) | 7.29–7.33 ms | 73.3–73.6 | 1.06–1.11× slower |
| **`weight_dequantization.cu`** (float4, 4 cells/thread) | **6.51–6.96 ms** | **77–82** | 1.0× |
| Numba CUDA (int32 + shift) | 7.06 ms | 76.1 | 1.01–1.08× slower |
| *float4 copy floor* | *6.55–6.72 ms* | *80–82* | *1.00×* |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test weight_dequantization.cu test_harness.cu
```

Numba:

```bash
python3 weight_dequantization.py
```

The script runs the correctness cases and the performance case.
