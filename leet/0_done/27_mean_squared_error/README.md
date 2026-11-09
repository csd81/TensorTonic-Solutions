# Mean Squared Error — CUDA C++ and Numba Results

Challenge: `mse = (1/N)·Σ (predictions[i] − targets[i])²`, floats,
`N ≤ 1e8`; benchmark **N = 50,000,000** — 400 MB of reads, so the kernel is
read-bound (the floor at this GPU's ~87 GB/s is ~4.6 ms).

Files:

- `mean_squared_error.cu` — CUDA C++ implementation (renamed from
  `starter.cu`).
- `mean_squared_error_naive.cu` — the naive baseline: the starter's kernel
  verbatim (all arithmetic in double, scalar loads).
- `mean_squared_error.py` — Numba CUDA port.

## Results at N = 50,000,000

Same-process interleaved medians:

| Variant | Time | Effective GB/s |
|---|---:|---:|
| **naive = starter kernel** | 5.64 ms (5.67 standalone) | 70.9 |
| **shipped** | **4.63 ms** | **86.4** |
| Numba CUDA port | 4.89 ms | 81.8 |

Speedup: **1.22×**, and the shipped kernel is at the memory wall.

## Why the starter leaves time on the table

It performs the inner-loop arithmetic in **double**: two converts, a subtract,
a multiply and an add per element.  On sm_61 FP64 runs at 1/32 of the FP32
rate, so those 250 M double operations cost more than the 400 MB stream.
The loads are also scalar 4-byte reads.

## Implementation — CUDA C++

1. **float4 loads** of both inputs (a warp request is four 512-byte
   transactions) with four independent FP32 accumulators.
2. **FP32 block reduction** via warp shuffles into one partial per block.
3. **FP64 only where it is free**: a single-block final kernel sums the
   ≤ 2048 block partials in double and divides by N.

Precision: each thread sums ~190 terms in FP32 and each block combines 256 of
those, keeping the accumulated relative error around 1e-6 — two orders below
the harness tolerance (measured error at the benchmark shape 6.7e-3 absolute
on 3.3e5, i.e. 2e-8 relative).

Correctness: both challenge examples and `N = 1, 7, 1000, 1e5, 2^20,
50,000,000` against a float64 reference (tolerance `1e-4·|ref| + 1e-3`).

A scalar tail handles `N mod 4` (≤ 3 elements) in thread 0 of block 0.

## Numba port

Same two-stage structure with scalar block-strided loads (Numba has no
float4), FP32 accumulation and warp/block reductions; the final kernel sums
the block partials in float64 (Numba supports `float64` shared arrays) and
divides by N.  Measured 4.89 ms (81.8 GB/s), within 6% of the C++ kernel.
