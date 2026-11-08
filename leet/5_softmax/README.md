# Softmax — CUDA C++ and Numba Results

Challenge: compute the numerically stable softmax of a `float32` array,
using the max trick `exp(x_i - max(x)) / sum(exp(x_j - max(x)))`.

Files:

- `softmax.cu` — CUDA C++ implementation of `solve(input, output, N)`
  (renamed from `starter.cu`).
- `softmax.py` — Numba CUDA port with the same `solve` signature.

## Implementation

Both versions use five kernels:

1. per-block max via grid-stride accumulation + warp/block reduction,
2. final max over the block maxima,
3. `exp(x - max)` written to `output` + per-block sums,
4. final sum over the block sums,
5. normalize `output` by the global sum.

Launch configuration: 256 threads/block, grid capped at 1024 blocks.

### Bug fixed in the CUDA starter

The starter passed the same scratch buffer (`d_temp`) as both `d_max` and
`block_sums` to `exp_and_sum_kernel`:

- each block reads the global max from `d_temp[0]`,
- block 0 writes its partial sum back to `d_temp[0]`.

Blocks scheduled after block 0 could therefore read a partial sum instead of
the max, corrupting the softmax. The implementation now allocates separate
`d_max` and `d_sums` buffers. It also synchronizes before freeing them, and
the warp-shuffle loops are marked `#pragma unroll`.

The Numba port uses separate `d_max` / `d_sums` device arrays from the start.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, sm_61) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-08 |

Kernel times are medians of 20 runs after a warm-up, with input/output already
on the device. The timed region includes `cudaDeviceSynchronize()`.

## Correctness — CUDA C++

| Case | N | Max abs error | Relative error | Output sum |
|---|---:|---:|---:|---:|
| [1, 2, 3] | 3 | 7.20e-08 | 1.08e-07 | 0.999999925 |
| [-10, -5, 0, 5, 10] | 5 | 6.01e-08 | 6.05e-08 | 0.999999940 |
| single [5] | 1 | 0.0 | 0.0 | 1.000000000 |
| random | 1,000 | 1.77e-08 | 1.77e-08 | 1.000000018 |
| random | 100,000 | 1.27e-08 | 1.83e-08 | 0.999999975 |
| random | 500,000 | 4.91e-08 | 9.77e-08 | 1.000000101 |

## Correctness — Numba CUDA

| Case | N | Max abs error | Relative error | Output sum |
|---|---:|---:|---:|---:|
| [1, 2, 3] | 3 | 7.20e-08 | 1.08e-07 | 0.999999940 |
| [-10, -5, 0, 5, 10] | 5 | 6.01e-08 | 6.05e-08 | 0.999999940 |
| single [5] | 1 | 0.0 | 0.0 | 1.000000000 |
| random | 1,000 | 8.05e-08 | 9.20e-08 | 0.999999881 |
| random | 100,000 | 6.30e-08 | 6.35e-08 | 0.999999940 |
| random | 500,000 | 2.05e-08 | 3.17e-08 | 1.000000000 |

The host-array path (1000 random values) matched the reference with max
absolute error 2.37e-08.

## Performance — N = 500,000

Approximate moved bytes: `5 · N · 4 = 10 MB` (max read, exp read + write,
normalize read + write).

| Implementation | Median time | Approx. bandwidth |
|---|---:|---:|
| CUDA C++ | **0.164 ms** | **61.0 GB/s** |
| Numba CUDA | 0.554 ms | 18.1 GB/s |

The Numba version is about 3.4× slower, mainly because of the five kernel
launches and the extra Numba dispatch overhead.

## Build / run

CUDA C++ (the file exports `solve`, so compile it with a harness that
provides `main`):

```bash
nvcc -O3 -arch=sm_61 -o test softmax.cu test_harness.cu
```

Numba:

```bash
python3 softmax.py
```

The Python script runs the correctness cases and the performance case. For
best performance, pass Numba device arrays; host arrays incur per-kernel
copies.
