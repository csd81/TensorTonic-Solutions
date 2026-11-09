# Count 3D Array Element — CUDA C++ and Numba Results

Challenge: count the elements equal to `P` in an `N × M × K` int32 array.
`1 ≤ N, M, K ≤ 1000`, values in `[1, 100]`; performance is measured at
`500 × 500 × 500` (125,000,000 elements).

Files:

- `count_3d_array_element.cu` — CUDA C++ implementation of
  `solve(input, output, N, M, K, P)` (renamed from `starter.cu`).
- `count_3d_array_element_naive.cu` — the naive baseline: scalar grid-stride
  loop, one atomicAdd per thread with matches.
- `count_3d_array_element.py` — Numba CUDA port with the same signature.

## Where the time goes at 500³

The kernel reads 500 MB and writes 4 bytes, so it is purely **read-bound**.
A plain int4 read-over-the-array kernel measures **5.76–5.82 ms
(86–87 GB/s)**, and every correct variant is on that floor:

| Variant | Time | GB/s |
|---|---:|---:|
| read floor (int4 sum) | 5.76–5.82 ms | 86–87 |
| starter (scalar grid-stride, per-thread atomic) | 5.79 ms | 86.4 |
| naive (same shape) | 5.76–5.80 ms | 86–87 |
| **shipped (int4 + warp/block reduction)** | **5.78–5.80 ms** | **86–87** |

There is nothing to win beyond the memory wall here; the shipped kernel's
differences (four comparisons per load, one atomic per block instead of one
per thread, a scalar tail for totals not divisible by 4) are structural and
measure identical to the starter.

## Implementation — CUDA C++

`int4` loads in a grid-stride loop (1024 blocks × 256 threads), four equality
comparisons per load, a warp shuffle reduction plus a per-block shared-memory
reduction, and one `atomicAdd` per block.  The `total % 4` tail is handled
scalar in the same kernel.  `solve` zeroes the output with
`cudaMemsetAsync`.

### Naive baseline

`count_3d_array_element_naive.cu` is the plainest correct form — the
starter's shape: scalar grid-stride loop and one `atomicAdd` per thread that
found matches.  It measures **5.76–5.80 ms**, the same as the shipped kernel:
the challenge is a pure read of 500 MB, and both forms saturate it.

## Implementation — Numba

Scalar grid-stride kernel with a per-thread count and one atomicAdd per
thread; a tiny kernel zeroes the output (the workspace is reused across
calls).  It reaches **6.352 ms (78.7 GB/s)**, 10% behind the C++ kernel —
the expected codegen gap for a scalar read stream.  Kernels use
`cache=True`.

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
| challenge example 1 (2×2×3, P=1) | 5 ✓ |
| challenge example 2 (1×3×2, P=1) | 0 ✓ |
| 1×1×1 with P=1 and P=50 | exact |
| 10×20×30, 100×100×100, 999×1×1, 1×1000×61 | exact |
| benchmark shape 500×500×500 | 1,249,947 ✓ |

All 9 harness cases pass.

## Correctness — Numba CUDA

All 8 cases pass (both examples and six random shapes).

## Performance — 500 × 500 × 500

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| starter (scalar, per-thread atomic) | 5.79 ms | 86.4 | 1.00× |
| `count_3d_array_element_naive.cu` (same shape) | 5.76–5.80 ms | 86–87 | 1.00× |
| **`count_3d_array_element.cu`** (int4 + block reduce) | **5.78–5.80 ms** | **86–87** | 1.0× |
| Numba CUDA | 6.352 ms | 78.7 | 1.10× slower |
| *int4 read floor* | *5.76–5.82 ms* | *86–87* | *1.00×* |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test count_3d_array_element.cu test_harness.cu
```

Numba:

```bash
python3 count_3d_array_element.py
```

The script runs the correctness cases and the performance case.
