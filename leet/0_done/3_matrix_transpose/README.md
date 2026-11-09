# Matrix Transpose — CUDA C++ and Numba Results

Challenge: transpose a row-major `float32` matrix. Input is `rows × cols`;
output is `cols × rows`, with `output[col, row] = input[row, col]`.

Files:

- `matrix_transpose.cu` — CUDA C++ implementation of
  `solve(input, output, rows, cols)` (renamed from `starter.cu`).
- `matrix_transpose.py` — Numba CUDA port with the same `solve` signature.

## Implementation

**CUDA C++**

- Classic shared-memory transpose with a 32×32 input tile.
- 32×8 threads per block (256 threads); each thread loads and stores four
  elements (`j = 0, 8, 16, 24`).
- Shared tile is padded to 32×33 to remove bank conflicts.
- Global reads and writes are coalesced along 32 consecutive threads.
- Bounds checks handle arbitrary `rows` and `cols`.

**Numba CUDA**

- Same 32×32 tile / 32×8 thread scheme.
- `solve` accepts 2-D or flat arrays and host NumPy arrays or Numba device
  arrays (host copies handled by Numba).

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, sm_61) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-08 |

Kernel times are medians of 20 runs after a warm-up, with input/output
already on the device. The timed region includes `cudaDeviceSynchronize()`.
Three process runs were used to check stability.

## Correctness — CUDA C++

| rows | cols | Max abs error |
|---:|---:|---:|
| 2 | 3 | 0.000e+00 |
| 3 | 1 | 0.000e+00 |
| 17 | 33 | 0.000e+00 |
| 513 | 129 | 0.000e+00 |
| 129 | 513 | 0.000e+00 |

## Correctness — Numba CUDA

| rows | cols | Max abs error |
|---:|---:|---:|
| 2 | 3 | 0.000e+00 |
| 3 | 1 | 0.000e+00 |
| 17 | 33 | 0.000e+00 |
| 513 | 129 | 0.000e+00 |
| 129 | 513 | 0.000e+00 |
| 13 | 29 | 0.000e+00 (flat host and flat device paths) |

Transpose is a pure data movement operation, so the results are exact
(max absolute error zero).

## Performance — rows = 7,000, cols = 6,000

Data moved: `2 · rows · cols · 4 = 336 MB` (read + write).

| Implementation | Median time | Effective bandwidth |
|---|---:|---:|
| CUDA C++ — 16×16 tile, 256 threads | 6.816 ms | 49.3 GB/s |
| CUDA C++ — 32×32 tile, 1024 threads | 4.970 ms | 67.6 GB/s |
| **CUDA C++ — 32×8 threads, 4 elems/thread (kept)** | **5.078 ms** | **66.2 GB/s** |
| Numba CUDA — 32×8 threads, 4 elems/thread | 5.138 ms | 65.4 GB/s |

The 32×32/1024-thread and 32×8/256-thread variants are within run-to-run
noise; the 32×8 version was kept because it uses smaller blocks and gives
the same bandwidth. The Numba port is within ~1% of the CUDA C++ kernel.

The Quadro P2000 has a theoretical memory bandwidth of about 140 GB/s; the
transpose reaches roughly 47% of it. Individual run medians varied by a few
percent (C++: 5.03–5.16 ms, Numba: 5.07–5.25 ms).

## Build / run

CUDA C++ (the file exports `solve`, so compile it with a harness that
provides `main`):

```bash
nvcc -O3 -arch=sm_61 -o test matrix_transpose.cu test_harness.cu
```

Numba:

```bash
python3 matrix_transpose.py
```

The Python script runs the correctness cases and the performance case.

## Naive baseline

`matrix_transpose_naive.cu` is the same computation written as plainly as possible:
one thread per output element computing its source index directly (one read, one write, no shared tile) — no shared-memory tiling, no vector types, no warp intrinsics and no
fast-math.  It passes the same correctness cases as the optimized kernel
(checked against a float64 CPU reference in the shared harness).

| Implementation | Time | vs naive |
|---|---:|---:|
| `matrix_transpose_naive.cu` | 17.025 ms | 1.00x |
| `matrix_transpose.cu` (optimized) | 5.022 ms | 3.39x |

The optimized kernel tiles through shared memory so both the reads and the writes stay coalesced.
