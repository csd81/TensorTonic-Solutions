# Matrix Copy — CUDA C++ and Numba Results

Challenge: copy an `N × N` float matrix, `B[i][j] = A[i][j]`, with
`1 ≤ N ≤ 4096`; performance is measured at `N = 4096`.

Files:

- `matrix_copy.cu` — CUDA C++ implementation of `solve(A, B, N)` (renamed
  from `starter.cu`).
- `matrix_copy_naive.cu` — the naive baseline: one thread per element,
  scalar load and store.
- `matrix_copy.py` — Numba CUDA port with the same signature.

## Where the time goes at N = 4096

`N² = 16,777,216` elements = 64 MB in + 64 MB out = **128 MB** of traffic.
This kernel *is* a copy, so there is no separate copy floor to compare
against: the memory system is the ceiling, and at this GPU's ~82 GB/s
read+write rate that is **~1.63 ms**.

Every mapping below is fully coalesced.  Measured variants (medians of 50
launches through the same harness):

| Variant | Time | GB/s |
|---|---:|---:|
| **scalar, 256 thr** | **1.628–1.646 ms** | **~82** |
| scalar, 512 thr | 1.634–1.644 ms | ~82 |
| float4, 128 thr | 1.645–1.648 ms | ~81.5 |
| float4, 256 thr | 1.635–1.649 ms | ~81.5 |
| float4, 512 thr | 1.630–1.652 ms | ~82 |
| float4 with `__stcs` streaming stores | 1.634–1.644 ms | ~82 |
| float4 grid-stride, 512–1024 blocks | 1.653–1.671 ms | 80.3–81.2 |

Vectorizing buys nothing — a 16-byte load per thread saves instructions that
are not the bottleneck — and grid-stride shapes with fewer blocks are
slightly worse.  The shipped implementation is therefore the plain scalar
copy with 256-thread blocks: it is the fastest measured form, and it is also
the simplest, needing no alignment or tail special case.

## Implementation — CUDA C++

One thread per element, 256-thread blocks, `__ldg` on the source, one launch.
No vector types, no grid-stride loop, no alignment checks (scalar accesses
are valid for any pointer and any `N²`).

### Naive baseline

`matrix_copy_naive.cu` is the plainest correct form: one thread per element,
plain scalar load/store — structurally the same as the shipped kernel.  It
measures **1.635 ms (82.1 GB/s)**, identical to the final version within
noise.  That is the honest result for this challenge: copying is the
operation, so the naive mapping is the optimal mapping.  (This is also what
challenge 19, reverse-array, found for its swap kernel.)

## Implementation — Numba

Scalar one-element-per-thread copy — the same mapping, since there is nothing
to vectorize for and Numba has no `float4` type anyway.  It reaches
**1.719 ms (78.1 GB/s)**, 4.6% behind the C++ kernel; at 128 MB of traffic
the per-launch overhead is negligible, so the gap is codegen (bounds checks,
scalar loads).  Kernels use `cache=True`.

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
C++ variants were compiled against the same harness and run interleaved on an
idle GPU.  Run-to-run variation is about ±1%.

## Correctness — CUDA C++

The copy is exact, so every case is checked for bit-exact equality:

| Case | Result |
|---|---|
| challenge example 1 (2×2) | exact |
| challenge example 2 (3×3) | exact |
| N = 1 | exact |
| sizes 2, 5, 7, 31, 64, 100, 255, 1000, 4096 | exact |
| misaligned `A+1`, `B+1` (N = 1000) | exact |
| misaligned `A+2`, `B+3` (N = 1001) | exact |
| benchmark shape, N = 4096 | exact |

## Correctness — Numba CUDA

All cases pass bit-exact, including both challenge examples and N = 4096.

## Performance — N = 4096

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| `matrix_copy_naive.cu` (scalar, 1 thread/element) | 1.635 ms | 82.1 | 1.00× |
| starter (scalar, 1 thread/element) | 1.645 ms | 81.6 | 1.01× |
| **`matrix_copy.cu`** (scalar, 256-thread blocks) | **1.644 ms** | **81.7** | 1.0× |
| Numba CUDA | 1.719 ms | 78.1 | 1.05× slower |

Everything on the GPU side is the same number within noise: this is the
memory wall itself, and there is no algorithmic work to do.

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test matrix_copy.cu test_harness.cu
```

`THREADS` can be overridden with `-D` (the default is 256).

Numba:

```bash
python3 matrix_copy.py
```

The script runs the correctness cases and the performance case.
