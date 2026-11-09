# RMS Norm Backward — CUDA C++ and Numba Results

Challenge: RMS-norm backward for `[N, C]`:
`rstd_i = 1/√(mean(x²)+ε)`, `xhat = x·rstd`, `g = dy·w`,
`dx = rstd·(g − xhat·mean(g·xhat))`, `dweight_j = Σ_i dy_ij·xhat_ij`.
Benchmark: **N = 16,384, C = 4,096** — x, dy, dx are 268 MB each; one read of
x and dy plus one write of dx is 804 MB, ~9.8 ms at this GPU's ~82 GB/s.

Files:

- `rms_norm_backward.cu` — CUDA C++ implementation (renamed from
  `starter.cu`).
- `rms_norm_backward_naive.cu` — the naive baseline: the starter's kernel
  verbatim.
- `rms_norm_backward.py` — Numba CUDA port.

## Results at the benchmark shape

Same-process interleaved medians:

| Variant | Time | Effective GB/s | max dweight error |
|---|---:|---:|---:|
| **naive = starter kernel** | 21.67 ms (21.71 standalone) | 37.2 | 1.1e-2 |
| **shipped** | **10.53 ms** | **76.5** | **2.6e-4** |
| Numba CUDA port | 14.27 ms | 56.4 | 2.2e-4 |

Speedup: **2.06×**, at 93% of the 804 MB memory wall — and the shipped kernel
is ~40× more accurate on `dweight` than the starter.

## Why the starter leaves 2× (and precision) on the table

It reads x three times and dy twice (1.6 GB instead of 804 MB) and performs
one **float** `atomicAdd` per element for dweight — 67 M atomics whose
cancellation over 16 K rows leaves up to ~1e-2 absolute error on columns
whose true `dweight` is small.

## Implementation — CUDA C++

1. **Registers**: each block processes R = 16 consecutive rows; per row, x,
   dy and weight are held in per-thread registers (COLS = ceil(C/256) ≤ 16
   columns per thread, templated over 1/2/4/8/16), so the row is read once,
   dx is written once, and nothing is re-read.
2. **dweight**: the block's columns accumulate in registers across its R
   rows, then **one `atomicAdd(double)` per column per block** — 4.2 M double
   atomics instead of 67 M float ones, reducing the error to ~1e-7 relative
   (measured max 2.6e-4 absolute on values of magnitude ~2×10³).
3. Two block reductions per row (sumsq, then g·xhat) with warp shuffles plus
   a shared combine; the broadcast helper ends with a barrier because the
   reductions repeat for every row.

A generic kernel covers C > 4,096 (correctness only; the benchmark is 4,096).

Correctness: the challenge example, `1x1`, `1x7`, `5x3`, `17x33`, `1000x64`,
`3x1024`, `2x4096` and the benchmark shape; `dx` and `dweight` are checked
separately (`dx`: `1e-4 + 1e-4|ref|`; `dweight`: `1e-3 + 1e-4|ref|`).

## Numba port

Keeps the accuracy strategy without register blocking: R = 8 rows per block
accumulate dweight for the block's columns in a shared array (each column is
owned by one thread for the whole block, so the updates are race-free), then
one float64 atomic per column per block.  The row data is re-read from L1 for
the normalize pass (Numba cannot hold 16 columns in registers across a
reduction without spilling).  Measured 14.27 ms (56.4 GB/s) with the same
dweight accuracy class as the C++ kernel.
