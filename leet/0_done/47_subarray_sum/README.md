# Subarray Sum — CUDA C++ and Numba Results

Challenge: sum `input[S..E]` (inclusive, 0-based), int32 values in `[1, 10]`.
`1 ≤ N ≤ 100,000,000`; performance is measured at `N = 100,000,000`.

Files:

- `subarray_sum.cu` — CUDA C++ implementation of `solve(input, output, N, S, E)`
  (renamed from `starter.cu`).
- `subarray_sum_naive.cu` — the naive baseline: scalar grid-stride loop with
  a shared-memory tree reduction.
- `subarray_sum.py` — Numba CUDA port with the same signature.

## Where the time goes at N = 100,000,000

A full-range sum reads 400 MB and writes 4 bytes, so it is read-bound.  A
pure int4 read of the same array measures ~4.65 ms (86 GB/s):

| Variant | Time (full range) | GB/s |
|---|---:|---:|
| starter (scalar grid-stride, shared tree) | 5.09 ms | 78.5 |
| naive (same shape) | 4.95–5.02 ms | 80–81 |
| **shipped (int4 body, 4096 blocks)** | **4.61–4.63 ms** | **86.3–86.8** |
| Numba CUDA (scalar) | 4.90 ms | 81.6 |

The starter issues one 4-byte load per thread per step; the shipped kernel
loads `int4` (four values per instruction), which is worth ~8% on a pure read
stream.  Partial ranges behave the same (4.61–4.62 ms for `S=1000,
E=N−1001`).

## Implementation — CUDA C++

The range is split into an aligned `int4` body and at-most-3 scalar elements
at each end:

- `i0` is the first index ≥ `S` that is a multiple of 4, `i1` is the largest
  index ≤ `E+1` such that `i1 − i0` is a multiple of 4.  The body
  `[i0, i1)` uses `int4` loads; the head `[S, i0)` and tail `[i1, E+1]` are
  summed by thread 0 of block 0 (≤ 6 elements).
- Grid-stride loop over the body, warp-shuffle reduction plus a per-block
  shared reduction, one `atomicAdd` per block (only when non-zero).

Block-count sweep: 1024 → 4.64, 2048 → 4.62, **4096 → 4.61**, 8192 → 4.63 ms.

### Naive baseline

`subarray_sum_naive.cu` is the plainest correct form — the starter's shape:
scalar grid-stride loop, shared-memory tree reduction, one atomic per block.
It measures **4.95–5.02 ms**, ~8% slower.

## Implementation — Numba

Scalar grid-stride kernel with a per-thread sum and one atomicAdd per thread
that saw elements (Numba has no int4 type); a tiny kernel zeroes the output.
It reaches **4.90 ms (81.6 GB/s)**, in line with the C++ scalar form.
Kernels use `cache=True`.

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
C++ variants were compiled against the same harness and run interleaved.

## Correctness — CUDA C++

Exact integer results, checked against a 64-bit CPU sum:

| Case | Result |
|---|---|
| challenge example 1 (`[1,2,1,3,4]`, S=1, E=3) | 6 ✓ |
| challenge example 2 (`[1,2,3,4]`, S=0, E=3) | 10 ✓ |
| single element, single mid/last element | exact |
| full 1000; partial 10..900; full 1M; partial 123456..654321 | exact |
| benchmark full (100 M) and partial (1000..N−1001) | exact |

All 11 harness cases pass.

## Correctness — Numba CUDA

All 9 cases pass, exact.

## Performance — N = 100,000,000

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| starter (scalar) | 5.09 ms | 78.5 | 1.10× slower |
| `subarray_sum_naive.cu` (same shape) | 4.95–5.02 ms | 80–81 | 1.07–1.09× slower |
| **`subarray_sum.cu`** (int4) | **4.61–4.63 ms** | **86.3–86.8** | 1.0× |
| Numba CUDA | 4.90 ms | 81.6 | 1.06× slower |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test subarray_sum.cu test_harness.cu
```

`MAX_BLOCKS` can be overridden with `-D` (the default is 4096).

Numba:

```bash
python3 subarray_sum.py
```

The script runs the correctness cases and the performance case.
