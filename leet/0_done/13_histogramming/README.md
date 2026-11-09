# Histogramming — CUDA C++ and Numba Results

Challenge: count occurrences of each value in an array of 32-bit integers
with values in `[0, num_bins)`.  `1 ≤ N ≤ 100,000,000`, `1 ≤ num_bins ≤
1,024`; performance is measured at `N = 50,000,000`, `num_bins = 256`.

Files:

- `histogramming.cu` — CUDA C++ implementation of
  `solve(input, histogram, N, num_bins)` (renamed from `starter.cu`).
- `histogramming.py` — Numba CUDA port with the same signature.

## The starter was already at the roofline

The histogram itself is 256 integers; the whole cost is reading 200 MB once.
At this size the starter measured **2.29–2.31 ms (86–87 GB/s)**, and a pure
read-only kernel with the same `int4` grid-stride shape measures the same:

| Kernel (N = 50 M, 256 bins) | Time | Bandwidth |
|---|---:|---:|
| read-only sum, `int4` (measured ceiling) | 2.291–2.304 ms | 86.9–87.3 GB/s |
| read-only with `__ldcs` streaming loads | 2.304 ms | 86.8 GB/s |
| read-only, two `int4` per thread | 2.291 ms | 87.3 GB/s |
| **starter: per-block shared histogram + atomics** | **2.303–2.317 ms** | **86.1–86.8 GB/s** |
| per-warp privatised histograms (8 × 256 bins) | 2.303–2.314 ms | 86.4–86.8 GB/s |

The 50 million shared-memory `atomicAdd`s cost nothing: they hide completely
behind the DRAM latency, and spreading them over eight per-warp histograms
(8× fewer conflicts) changes nothing either.  There is no algorithmic
headroom — every element must be read once, and the input is 32-bit so it
cannot be packed narrower.

The final file therefore keeps the block-histogram kernel and adds only:

- the SM count cached instead of queried on every call,
- an alignment check with a scalar fallback (the `int4` fast path assumes a
  16-byte aligned input),
- `cudaMemsetAsync` for the output and a grid capped by both the SM count and
  the problem size.

## Implementation — Numba

The same block-histogram kernel with a 1024-entry shared array (the maximum
bin count) and a grid-stride loop.  Numba has no `int4` loads, so it reads
scalar 32-bit words: 2.62 ms (76 GB/s) versus the C++ kernel's 2.29 ms.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, 6 SMs) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Times are medians of 20 runs after a warm-up, including
`cudaDeviceSynchronize()`.

## Correctness — CUDA C++

Compared against a host count; every case matches exactly:

| Case | N | bins |
|---|---:|---:|
| challenge example 1 (`[0,1,2,1,0]` → `[2,2,1]`) | 5 | 3 |
| challenge example 2 (`[3,3,3,3]` → `[0,0,0,4,0]`) | 4 | 5 |
| single element | 1 | 1 |
| one bin | 4 | 1 |
| bins with zero counts | 7 | 8 |
| small random | 1,000 | 3 |
| random | 1,000 | 256 |
| random (tail of the `int4` path exercised) | 1,000,001 | 256 |
| large k | 1,048,576 | 1,024 |
| all values identical | 10,000 | 256 |
| benchmark size | 50,000,000 | 256 |

## Correctness — Numba CUDA

All ten cases match `np.bincount` exactly.

## Performance — N = 50,000,000, num_bins = 256

| Implementation | Time | Bandwidth |
|---|---:|---:|
| starter | 2.30 ms | 86.7 GB/s |
| **CUDA C++ final** | **2.29 ms** | **87.4 GB/s** |
| Numba CUDA | 2.62 ms | 76.4 GB/s |

The C++ kernel is within noise of the starter and of a pure read — the
histogramming is free at this size.  (An alternating head-to-head measures
2.310/2.347 ms for the final kernel against 2.307/2.336 ms for the starter,
with identical minima of ~2.274 ms.)

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test histogramming.cu test_harness.cu
```

`HIST_THREADS` and `HIST_BLOCKS_PER_SM` can be overridden with `-D` (the
defaults are 256 and 32).

Numba:

```bash
python3 histogramming.py
```

The script runs the correctness cases and the performance case.  For best
performance, pass Numba device arrays; host arrays incur copies to and from
the device.
