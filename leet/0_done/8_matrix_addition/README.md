# Matrix Addition — CUDA C++ and Numba Results

Challenge: element-wise addition of two `N×N` `float32` matrices,
`C = A + B`, with `1 ≤ N ≤ 4096`.  Performance is measured with `N = 4096`
(three 64 MiB matrices).

Files:

- `matrix_addition.cu` — CUDA C++ implementation of
  `solve(A, B, C, N)` (renamed from `starter.cu`).
- `matrix_addition.py` — Numba CUDA port with the same `solve` signature.

## Implementation

This is a pure streaming problem: every call reads `A` and `B` once and
writes `C` once — 192 MiB of mandatory DRAM traffic at `N = 4096`.  The
final kernel:

- loads `float4` vectors (16 bytes) from `A` and `B` and stores one `float4`
  to `C`, so each thread performs two 128-bit loads and one 128-bit store
  instead of the starter's three scalar accesses per element;
- 256 threads per block, one `float4` per thread;
- falls back to a scalar kernel if any of the three pointers is not 16-byte
  aligned, and handles the at-most-three leftover elements when `N*N` is not
  a multiple of four (odd `N`) in thread 0 of block 0 in the same launch.

### Result: the problem is bandwidth-bound, so vectorising gains ~nothing

All tested configurations land within ±1% of each other:

| Variant | Time | Bandwidth |
|---|---:|---:|
| starter (scalar, 1 element/thread) | 2.400 ms | 83.9 GB/s |
| `float4` ×1, 256 threads (final) | 2.403–2.423 ms | 83.1–83.8 GB/s |
| `float4` ×1, 128 threads | 2.407 ms | 83.7 GB/s |
| `float4` ×1, 512 threads | 2.407 ms | 83.6 GB/s |
| `float4` ×2 per thread | 2.438 ms | 82.6 GB/s |
| streaming loads/stores (`__ldcs`/`__stcs`) | 2.417 ms | 83.3 GB/s |
| scalar, measured in the same process as `float4` | 2.418 ms | 83.3 GB/s |

The GPU's observed practical DRAM ceiling is ~84 GB/s (the vector-add and
reduction challenges reach the same number), and this kernel already sits at
it.  The final version is kept because it issues 4× fewer memory
instructions and handles odd `N` and unaligned buffers explicitly, not
because it is measurably faster.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, sm_61) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-08 |

Kernel times are medians of 30 runs (C++) / 20 runs (Numba) after a warm-up,
with inputs already on the device, and include `cudaDeviceSynchronize()`.

## Correctness

`float32` addition is exact-rounded, so the results are compared bitwise and
match the CPU reference with zero mismatches on every case:

| Case | N | Elements |
|---|---:|---:|
| challenge example 1 | 2 | 4 |
| challenge example 2 | 3 | 9 |
| single element | 1 | 1 |
| odd | 5 | 25 |
| odd | 7 | 49 |
| odd | 33 | 1,089 |
| square | 100 | 10,000 |
| odd | 255 | 65,025 |
| square | 256 | 65,536 |
| square | 1000 | 1,000,000 |
| odd | 1023 | 1,046,529 |
| benchmark size | 4096 | 16,777,216 |

The challenge examples produce `[[6, 8], [10, 12]]` and
`[[2, 3, 4], [5, 6, 7], [8, 9, 10]]` exactly.  The CUDA C++ version was also
tested with all three pointers offset by one float (4 bytes), exercising the
scalar fallback, and the benchmark output was re-verified after timing.

The Numba port passes the same suite for both host and device arrays.

## Performance — N = 4096

Traffic per call: `3 · N² · 4 B = 192 MiB`.

| Implementation | Median time | Bandwidth |
|---|---:|---:|
| CUDA C++ final | 2.403–2.423 ms | 83.1–83.8 GB/s |
| Numba CUDA | 2.453 ms | 82.1 GB/s |

Both implementations are at the DRAM bandwidth limit; the ~1–2% gap is noise
plus Numba dispatch overhead.

## Build / run

CUDA C++ (the file exports `solve`, so compile it with a harness that
provides `main`):

```bash
nvcc -O3 -arch=sm_61 -o test matrix_addition.cu test_harness.cu
```

Numba:

```bash
python3 matrix_addition.py
```

The Python script runs the correctness cases and the performance case.  For
best performance, pass Numba device arrays; host arrays incur copies to and
from the device.

## Naive baseline

`matrix_addition_naive.cu` is the same computation written as plainly as possible:
one thread per element with a plain scalar add — no shared-memory tiling, no vector types, no warp intrinsics and no
fast-math.  It passes the same correctness cases as the optimized kernel
(checked against a float64 CPU reference in the shared harness).

| Implementation | Time | vs naive |
|---|---:|---:|
| `matrix_addition_naive.cu` | 2.409 ms | 1.00x |
| `matrix_addition.cu` (optimized) | 2.416 ms | 1.00x |

Both are bandwidth-bound at the same 192 MiB of traffic; vectorising does not help on this GPU.
