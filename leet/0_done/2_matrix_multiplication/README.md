# Matrix Multiplication — CUDA C++ and Numba Results

Challenge: compute `C = A × B` for row-major `float32` matrices
`A (M × N)`, `B (N × K)`, `C (M × K)`.

Files:

- `matrix_multiplication.cu` — CUDA C++ implementation of `solve(A, B, C, M, N, K)`.
- `matrix_multiplication.py` — Numba CUDA port with the same `solve` signature.

## Implementation

Both versions use the same register-blocked tiling scheme:

- 16 × 16 threads per block (256 threads).
- One block computes a **64 × 64** output tile.
- Each thread computes a **4 × 4** sub-tile using 16 register accumulators.
- `A` tile **64 × 16** and `B` tile **16 × 64** are staged in shared memory.
- Bounds checks handle arbitrary `M`, `N`, `K`.
- C++ uses `fmaf`; the Numba version carries 16 scalar accumulators.
- `solve` writes the result into `C` and keeps the required signature.

The Numba port accepts 2-D or flat arrays and host NumPy arrays or Numba
device arrays (Numba handles host copies).

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, sm_61) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-08 |

GPU kernel times were measured with `cudaEvent`/`cuda.synchronize()`; input
data was already on the device, so the times exclude H2D/D2H transfers.
Correctness used a float64 CPU reference and float32 GPU results.

## Correctness — CUDA C++

| M | N | K | Max abs error | Relative error |
|---:|---:|---:|---:|---:|
| 2 | 2 | 2 | 2.184e-08 | 2.184e-08 |
| 1 | 3 | 1 | 9.277e-08 | 5.366e-08 |
| 17 | 33 | 19 | 4.077e-06 | 2.181e-07 |
| 128 | 257 | 193 | 3.593e-05 | 5.304e-07 |
| 513 | 129 | 257 | 2.308e-05 | 4.573e-07 |

## Correctness — Numba CUDA

| M | N | K | Max abs error | Relative error |
|---:|---:|---:|---:|---:|
| 2 | 2 | 2 | 5.532e-09 | 5.532e-09 |
| 1 | 3 | 1 | 6.150e-09 | 2.017e-09 |
| 17 | 33 | 19 | 2.717e-06 | 1.609e-07 |
| 128 | 257 | 193 | 5.330e-05 | 7.541e-07 |
| 64 | 96 | 48 | 1.200e-05 | — |

The 64×96×48 Numba case was also run with flat host arrays and flat device
arrays; all three paths produced the same result (max abs error 1.200e-05).

## Performance — M=8192, N=6144, K=4096

Input matrices filled with ones; the GPU result was spot-checked against the
expected value `N = 6144`.

| Implementation | Kernel time | Approx. throughput |
|---|---:|---:|
| CUDA C++ — simple 16×16 tile (1 output/thread) | 1765.16 ms | 234 GFLOP/s |
| CUDA C++ — 64×64 register-blocked (4×4/thread) | **578.74 ms** | **712 GFLOP/s** |
| Numba CUDA — 64×64 register-blocked (4×4/thread) | 1146.2 ms | 360 GFLOP/s |

The workload is `2·M·N·K ≈ 412.3 GFLOP`. The register-blocked CUDA kernel is
about 3× faster than the simple tiled version and about 2× faster than the
Numba port.

## Build / run

CUDA C++ (the file is a challenge library with `solve`, so compile it with a
test harness that provides `main`):

```bash
nvcc -O3 -arch=sm_61 -o test matrix_multiplication.cu test_harness.cu
```

Numba:

```bash
python3 matrix_multiplication.py
```

The script runs the small correctness cases listed above.

## Naive baseline

`matrix_multiplication_naive.cu` is the same computation written as plainly as possible:
one thread per element of C walking the whole K reduction with plain global-memory loads — no shared-memory tiling, no vector types, no warp intrinsics and no
fast-math.  It passes the same correctness cases as the optimized kernel
(checked against a float64 CPU reference in the shared harness).

| Implementation | Time | vs naive |
|---|---:|---:|
| `matrix_multiplication_naive.cu` | ~17,200 ms | 1.00x |
| `matrix_multiplication.cu` (optimized) | 561.9 ms | ~31x |

The naive figure is approximate: it was measured once while the GPU was shared with other work and was not re-timed.
