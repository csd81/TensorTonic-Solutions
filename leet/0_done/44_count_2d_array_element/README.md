# Count 2D Array Element — CUDA C++ and Numba Results

Challenge: count the elements equal to `k` in an `N × M` int32 array.
`1 ≤ N, M ≤ 10,000`, values in `[1, 100]`; performance is measured at
`10,000 × 10,000` (100,000,000 elements).

Files:

- `count_2d_array_element.cu` — CUDA C++ implementation of
  `solve(input, output, N, M, K)` (renamed from `starter.cu`).
- `count_2d_array_element_naive.cu` — the naive baseline: scalar grid-stride
  loop with a shared-memory tree reduction.
- `count_2d_array_element.py` — Numba CUDA port with the same signature.

## Where the time goes at 10,000²

The kernel reads 400 MB and writes 4 bytes, so it is purely **read-bound**.
The starter's scalar grid-stride kernel already measures **4.61 ms
(86.7 GB/s)** — at the read ceiling — and every correct variant is there:

| Variant | Time | GB/s |
|---|---:|---:|
| starter (scalar, shared tree) | 4.61 ms | 86.7 |
| naive (same shape) | 4.60–4.62 ms | 86.5–86.9 |
| **shipped (int4 + warp/block reduction)** | **4.62–4.68 ms** | **85.5–86.5** |
| Numba CUDA (scalar) | 4.794 ms | 83.4 |

There is no headroom beyond the memory wall; the shipped kernel's int4 loads
and one-atomic-per-block reduction measure the same as the scalar starter.

## Implementation — CUDA C++

`int4` loads in a grid-stride loop (up to 4096 blocks × 256 threads), four
equality comparisons per load, a scalar tail for totals not divisible by 4, a
warp shuffle plus per-block shared reduction, and one `atomicAdd` per block.
`solve` zeroes the output with `cudaMemsetAsync`.

### Naive baseline

`count_2d_array_element_naive.cu` is the plainest correct form — the
starter's shape: scalar grid-stride loop, shared-memory tree reduction, one
atomic per block.  It measures **4.60–4.62 ms**, the same as the shipped
kernel.

## Implementation — Numba

Scalar grid-stride kernel with a per-thread count and one atomicAdd per
thread; a tiny kernel zeroes the output.  It reaches **4.794 ms
(83.4 GB/s)**.  Kernels use `cache=True`.

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

Exact integer results, checked against a CPU count:

| Case | Result |
|---|---|
| challenge example 1 (2×3, k=1) | 2 ✓ |
| challenge example 2 (2×2, k=1) | 0 ✓ |
| 1×1 with k=1 and k=50 | exact |
| 10×20, 100×100, 1×1,000,000, 999×1001, 10,000×10,000 | exact |

All 9 harness cases pass.

## Correctness — Numba CUDA

All 8 cases pass, exact.

## Performance — 10,000 × 10,000

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| starter (scalar) | 4.61 ms | 86.7 | 1.00× |
| `count_2d_array_element_naive.cu` (same shape) | 4.60–4.62 ms | 86.5–86.9 | 1.00× |
| **`count_2d_array_element.cu`** (int4) | **4.62–4.68 ms** | **85.5–86.5** | 1.0× |
| Numba CUDA | 4.794 ms | 83.4 | 1.04× slower |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test count_2d_array_element.cu test_harness.cu
```

`MAX_BLOCKS` can be overridden with `-D` (the default is 4096).

Numba:

```bash
python3 count_2d_array_element.py
```

The script runs the correctness cases and the performance case.
