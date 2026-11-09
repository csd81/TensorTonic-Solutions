# Parallel Reduction (Sum) — CUDA C++ and Numba Results

Challenge: sum an array of `float32` values on the GPU and store the result
in `output[0]`.

Files:

- `reduction.cu` — CUDA C++ implementation of `solve(input, output, N)`
  (renamed from `starter.cu`).
- `reduction.py` — Numba CUDA port with the same `solve` signature.

## Implementation

Both versions use the same three-level reduction:

1. **Grid-stride loop** — each thread accumulates multiple elements.
2. **Warp reduction** — `__shfl_down_sync` / `cuda.shfl_down_sync`.
3. **Block reduction** — one partial per warp through a 32-element shared
   array, then a final warp shuffle; thread 0 performs one `atomicAdd` per
   block.

Launch configuration: 256 threads per block, grid capped at 1024 blocks.

The CUDA starter already contained the reduction structure. It was validated,
given an explicit `cudaDeviceSynchronize()`, and the warp-shuffle loop was
marked `#pragma unroll`.

In Numba, `output[0] = 0.0` caused a synchronous host-to-device scalar copy
(~0.27 ms per call), so the output is cleared with a one-thread
`zero_kernel` instead. The CUDA version uses `cudaMemset`.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, sm_61) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-08 |

Kernel times are medians of 20 runs after a warm-up, with the input already
on the device. The timed region includes `cudaDeviceSynchronize()`.
Correctness was checked against a float64 CPU sum.

## Correctness — CUDA C++

| N | Relative error vs CPU |
|---:|---:|
| 8 (example 1) | 0.0 |
| 4 (example 2) | 0.0 |
| 1 | 0.0 |
| 1,000 | ~5.0e-09 |
| 1,000,003 | ~5.3e-10 |
| 4,194,304 | 1.6e-10 … 4.0e-10 |

The random-data errors vary slightly between runs because the block
`atomicAdd` order is non-deterministic in float32.

## Correctness — Numba CUDA

| N | Relative error vs CPU |
|---:|---:|
| 8 (example 1) | 0.0 |
| 4 (example 2) | 0.0 |
| 1 | 0.0 |
| 1,000 | 8.8e-10 |
| 1,000,003 | 3.3e-10 |
| 4,194,304 | 3.3e-11 |

The host-array path was also tested and matched the CPU reference.

## Performance — N = 4,194,304

Input bytes: `4 · N = 16.78 MB`.

| Implementation | Median time | Effective bandwidth |
|---|---:|---:|
| CUDA C++ | **0.200 ms** | **83.8 GB/s** |
| Numba CUDA | 0.297 ms | 56.5 GB/s |

The Quadro P2000 has a theoretical memory bandwidth of about 140 GB/s; the
CUDA reduction reaches roughly 60% of it. Numba is about 1.5× slower than the
hand-written CUDA version.

## Build / run

CUDA C++ (the file exports `solve`, so compile it with a harness that
provides `main`):

```bash
nvcc -O3 -arch=sm_61 -o test reduction.cu test_harness.cu
```

Numba:

```bash
python3 reduction.py
```

The Python script runs the correctness cases and the performance case.

## Naive baseline

`reduction_naive.cu` is the same computation written as plainly as possible:
one thread per element with an atomicAdd straight into output[0] — no shared-memory tiling, no vector types, no warp intrinsics and no
fast-math.  It passes the same correctness cases as the optimized kernel
(checked against a float64 CPU reference in the shared harness).

| Implementation | Time | vs naive |
|---|---:|---:|
| `reduction_naive.cu` | 9.8396 ms | 1.00x |
| `reduction.cu` (optimized) | 0.2150 ms | 45.8x |

Serialising every element through one atomic is the difference; the optimized kernel reduces per block and adds once per block.
