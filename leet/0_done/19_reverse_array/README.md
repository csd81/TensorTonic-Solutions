# Reverse Array — CUDA C++ and Numba Results

Challenge: reverse an array of 32-bit floats **in place**.  `1 ≤ N ≤
100,000,000`; performance is measured at `N = 25,000,000`.

Files:

- `reverse_array.cu` — CUDA C++ implementation of `solve(input, N)` (renamed
  from `starter.cu`).
- `reverse_array.py` — Numba CUDA port with the same signature.

## The starter was already at the roofline

This is a pure streaming problem: every element is read once and written
once, so the kernel moves 200 MB at N = 25 M and there is nothing to save
algorithmically.  The starter measured **2.42 ms — 82.7 GB/s**, which is
~98% of this GPU's practical DRAM ceiling (83.7 GB/s measured with a
dedicated streaming kernel in the segmented-prefix-sum challenge).

Its access pattern gives nothing away either.  A warp's low-side read covers
32 ascending floats and the mirror-side read covers 32 descending floats
within the same 128-byte line; both are single fully-coalesced transactions,
and the same line is read and written once.

So the work here was to verify correctness and then confirm that nothing
faster exists — the measurements below were all taken in one process at the
benchmark size:

| Variant | Median | Bandwidth |
|---|---:|---:|
| scalar pair swap, 256 threads | 2.416 ms | 82.8 GB/s |
| scalar pair swap, 512 threads | **2.414 ms** | **82.9 GB/s** |
| `float4` vectors (512 B per warp) | 2.435 ms | 82.1 GB/s |
| `float4` + `__stcs` streaming stores | 2.429 ms | 82.3 GB/s |
| two `float4` per thread | 2.446 ms | 81.8 GB/s |

Vectorising does not help — the kernel is bandwidth-bound, not
instruction-bound — and streaming stores do not help because there is no
reuse to protect.  The final file therefore keeps the scalar pair swap (the
fastest of the set), with a 512-thread block, `__restrict__`, and an early
return for `N ≤ 1`.

The honest summary is that the starter was already optimal; the only
remaining differences are run-to-run noise (the final binary measures
2.423/2.428 ms in an alternating head-to-head against the starter's
2.419/2.421 ms).

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

Reference: `std::reverse` on a host copy.  Every case matches exactly:

| Case | N | Result |
|---|---:|---|
| challenge example 1 (`[1,2,3,4]` → `[4,3,2,1]`) | 4 | exact |
| challenge example 2 (`[1.5,2.5,3.5]` → `[3.5,2.5,1.5]`) | 3 | exact |
| single element | 1 | exact |
| two elements | 2 | exact |
| all-equal | 7 | exact |
| distinct values | 255, 256, 257, 1000, 1001 | exact |
| random | 1,048,576 | exact |
| benchmark size (sampled comparison against the input) | 25,000,000 | exact |

## Correctness — Numba CUDA

All thirteen cases match as well.

## Performance — N = 25,000,000

| Implementation | Time | Bandwidth |
|---|---:|---:|
| starter | 2.42 ms | 82.7 GB/s |
| **CUDA C++ final** | **2.43 ms** | **82.4 GB/s** |
| Numba CUDA | 2.46 ms | 81.3 GB/s |

All three are the same kernel at the same wall: at 82 GB/s the 200 MB take
~2.4 ms and the arithmetic is free.  (Numba's per-launch overhead is ~60 µs,
which is negligible against a 2.4 ms kernel.)

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test reverse_array.cu test_harness.cu
```

`REVERSE_THREADS` can be overridden with `-D` (the default is 512).

Numba:

```bash
python3 reverse_array.py
```

The script runs the correctness cases and the performance case.  For best
performance, pass Numba device arrays; host arrays incur copies to and from
the device.

## Naive baseline

`reverse_array_naive.cu` is the same computation written as plainly as possible:
one thread per mirror pair doing a scalar swap — no shared-memory tiling, no vector types, no warp intrinsics and no
fast-math.  It passes the same correctness cases as the optimized kernel
(checked against a float64 CPU reference in the shared harness).

| Implementation | Time | vs naive |
|---|---:|---:|
| `reverse_array_naive.cu` | 2.442 ms | 1.00x |
| `reverse_array.cu` (optimized) | 2.419 ms | 1.01x |

Both are at the DRAM wall (200 MB at N = 25 M); vectorising the accesses does not help.
