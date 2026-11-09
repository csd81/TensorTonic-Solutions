# Vector Addition — CUDA C++ and Numba Results

Challenge: compute `C[i] = A[i] + B[i]` for two `float32` vectors of length
`N`, storing the result in `C`.

Files:

- `vector_add.cu` — CUDA C++ implementation of `solve(A, B, C, N)` (renamed
  from `starter.cu`).
- `vector_add.py` — Numba CUDA port with the same `solve` signature.

## Implementation

**CUDA C++**

- One thread per element; 256 threads per block.
- Coalesced `float` loads and stores, bounds check `i < N`.
- The `solve` signature and launch configuration from the starter are kept.

A `float4`-vectorized variant (four elements per thread, scalar tail) was also
measured; it did not improve throughput on this GPU, so the simpler scalar
kernel is kept.

**Numba CUDA**

- The same one-thread-per-element kernel with 256 threads/block.
- `solve` accepts flat or 2-D arrays and host NumPy arrays or Numba device
  arrays (Numba handles host copies).

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, sm_61) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-08 |

Performance numbers are medians of 20 kernel runs after a warm-up, with the
input vectors already on the device (no H2D/D2H in the timed region). The
timed region includes the `cudaDeviceSynchronize()` in `solve`.

## Correctness — CUDA C++

| N | Max abs error | Relative error |
|---:|---:|---:|
| 4 | 0.000e+00 | 0.000e+00 |
| 1 | 5.309e-08 | 5.149e-08 |
| 1,000 | 1.192e-07 | 2.541e-08 |
| 1,000,003 | 2.384e-07 | 2.845e-08 |

## Correctness — Numba CUDA

| N | Max abs error | Relative error |
|---:|---:|---:|
| 4 | 5.960e-08 | 3.065e-08 |
| 1 | 0.000e+00 | 0.000e+00 |
| 1,000 | 2.384e-07 | 4.503e-08 |
| 1,000,003 | 2.384e-07 | 3.331e-08 |

## Performance — N = 25,000,000

Bytes moved: `3 · N · 4 = 300 MB` (read A, read B, write C).

| Implementation | Median kernel time | Effective bandwidth |
|---|---:|---:|
| CUDA C++ — scalar (kept version) | **3.573 ms** | **84.0 GB/s** |
| CUDA C++ — `float4` variant | 3.627 ms | 82.7 GB/s |
| Numba CUDA — scalar | 3.713 ms | 80.8 GB/s |

The kernel is memory-bandwidth-bound. The Quadro P2000 has a theoretical
memory bandwidth of about 140 GB/s; all three variants reach roughly 60% of
it. Single-run timings varied between ~3.6 and ~4.5 ms, so medians are
reported. The Numba version is within ~4% of the CUDA C++ version.

## Build / run

CUDA C++ (the file exports `solve`, so compile it with a harness that
provides `main`):

```bash
nvcc -O3 -arch=sm_61 -o test vector_add.cu test_harness.cu
```

Numba:

```bash
python3 vector_add.py
```

The Python script runs the correctness cases listed above.

## Naive baseline

`vector_add_naive.cu` is the same computation written as plainly as possible:
one thread per element reading A and B from global memory and writing C — no shared-memory tiling, no vector types, no warp intrinsics and no
fast-math.  It passes the same correctness cases as the optimized kernel
(checked against a float64 CPU reference in the shared harness).

| Implementation | Time | vs naive |
|---|---:|---:|
| `vector_add_naive.cu` | 3.594 ms | 1.00x |
| `vector_add.cu` (optimized) | 3.578 ms | 1.00x |

Both forms sit at the same DRAM wall: at N = 25 M the kernel moves 300 MB and nothing about the access pattern can change that.
