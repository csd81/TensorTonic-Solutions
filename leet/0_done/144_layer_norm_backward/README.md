# Layer Norm Backward — CUDA C++ and Numba Results

Challenge: layer-norm backward given the forward pass's `mean`/`rstd`:
`dx = rstd·(g − mean(g) − xhat·mean(g·xhat))`, `dgamma_j = Σ_i dy·xhat`,
`dbeta_j = Σ_i dy` with `g = dy·gamma`, `xhat = (x − mean)·rstd`.
Benchmark: **N = 16,384, C = 4,096** — x, dy, dx are 268 MB each; one read of
x and dy plus one write of dx is 804 MB, ~9.8 ms at ~82 GB/s.

Files:

- `layer_norm_backward.cu` — CUDA C++ implementation (renamed from
  `starter.cu`).
- `layer_norm_backward_naive.cu` — the naive baseline: the starter's two
  kernels verbatim.
- `layer_norm_backward.py` — Numba CUDA port.

## Results at the benchmark shape

Same-process interleaved medians:

| Variant | Time | Effective GB/s | max dgamma/dbeta error |
|---|---:|---:|---:|
| **naive = starter kernels** | 25.84 ms (24.54 standalone) | 31.2 | 1.5e-2 |
| **shipped** | **10.68 ms** | **75.4** | **2.2e-4** |
| Numba CUDA port | 28.63 ms | 28.1 | 1.9e-4 |

Speedup: **2.42×**, at 92% of the memory wall, and ~70× more accurate on the
parameter gradients.

## Why the starter leaves 2.4× (and precision) on the table

- Its **dx kernel** reads x and dy twice (once for the row sums, once for the
  normalize pass).
- Its **parameter kernel** re-reads x and dy a third time with **one thread
  per column**, summing all 16 K rows in a serial float32 loop.  Besides the
  extra traffic, only 4,096 threads (128 warps) are active for a 536 MB
  stream, and the serial float32 accumulation leaves up to ~1.5e-2 absolute
  error through cancellation.

## Implementation — CUDA C++

1. **Registers**: R = 16 consecutive rows per block; per row, x, dy and gamma
   are held in per-thread registers (COLS = ceil(C/256) ≤ 16, templated over
   1/2/4/8/16).  One read of x and dy, one write of dx, nothing re-read.
2. **dgamma/dbeta**: accumulated in registers over the block's R rows, then
   one `atomicAdd(double)` per column per parameter per block (4.2 M double
   atomics), which removes the serial-float accumulation error.
3. Two reductions per row (`sum(g)`, `sum(g·xhat)`) via warp shuffles plus a
   shared combine, with the post-broadcast barrier that back-to-back
   reductions require.

A generic kernel covers C > 4,096 (correctness only; the benchmark is 4,096).

Correctness: `1x1`, `1x7`, `5x3`, `17x33`, `1000x64`, `3x1024`, `2x4096` and
the benchmark shape against a float64 reference that uses the provided
float32 `mean`/`rstd` (as the kernel must).  `dx` tolerance
`1e-4 + 1e-4|ref|`, `dgamma/dbeta` `1e-3 + 1e-4|ref|`.

## Numba port

R = 8 rows per block; dgamma/dbeta accumulate in two shared arrays (each
column owned by one thread for the block, race-free) and then in float64
atomics per column per block — the same accuracy class (max 1.9e-4).  Row
data is re-read from L1 for the dx pass.  Measured 28.6 ms (28.1 GB/s): the
shared-array updates and scalar loads dominate.
