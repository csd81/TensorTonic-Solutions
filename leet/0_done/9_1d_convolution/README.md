# 1D Convolution (valid) — CUDA C++ and Numba Results

Challenge: compute the "valid" 1D convolution

```
out[i] = sum_{j < kernel_size} in[i + j] * k[j],   i < input_size - kernel_size + 1
```

for `1 ≤ input_size ≤ 1,500,000` and `1 ≤ kernel_size ≤ 2,047`, with
`kernel_size ≤ input_size`.  Performance is measured with
`input_size = 1,500,000`, `kernel_size = 2,047` (~3.07 G multiply-adds).

Files:

- `convolution_1d.cu` — CUDA C++ implementation of
  `solve(input, kernel, output, input_size, kernel_size)` (renamed from
  `starter.cu`).
- `convolution_1d.py` — Numba CUDA port with the same `solve` signature.

## Implementation — CUDA C++

The job is compute bound, so the kernel is built to issue as many FMAs as
possible per shared-memory access:

| Parameter | Value |
|---|---:|
| Threads per block | 640 |
| Outputs per thread | 8 (consecutive: `8t .. 8t+7`) |
| Outputs per block (`BLOCK_OUT`) | 5,120 |
| Shared memory | kernel 2,048 floats + input tile 7,170 floats |

Per block:

1. the kernel and the input tile (`BLOCK_OUT + kernel_size - 1` floats, with
   the right halo) are staged in shared memory; out-of-range inputs are
   zeroed;
2. each thread keeps its eight-element input window in registers (two
   `float4`s) and walks the taps four at a time:
   - one broadcast `float4` load brings taps `j..j+3`,
   - one `float4` load brings the four new input values (a warp reads 512
     contiguous bytes, conflict-free),
   - 32 FMAs are issued,
   - the window advances by four (`A0 = A1; A1 = B`);
3. a scalar tail handles `kernel_size % 4` leftover taps, and outputs past
   the valid region are masked.

This gives ~0.5 bytes of shared input traffic per MAC (the starter reads two
shared operands per MAC).

## Implementation — Numba

Numba has no vector shared-memory loads, so the port uses a different
mapping that is still conflict-free: thread `t` computes the strided outputs
`t, t+768, ..., t+7*768`, and one broadcast of `k[j]` feeds eight FMAs.  The
tap loop is manually unrolled by 8 (Numba does not unroll it), which is worth
~35% over the plain loop.  Shared memory: kernel 2,048 + tile 8,194 floats.

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
with inputs on the device, including `cudaDeviceSynchronize()`.  Correctness
is checked against a float64 CPU reference (full `np.correlate` for small
cases, sampled dot products for large ones).

## Correctness — CUDA C++

| Case | n | k | Outputs | Max abs error | Relative error |
|---|---:|---:|---:|---:|---:|
| challenge example 1 | 5 | 3 | 3 | 0.0 | 0.0 |
| challenge example 2 | 4 | 2 | 3 | 1.19e-07 | 2.59e-08 |
| single element | 1 | 1 | 1 | 1.49e-08 | 2.36e-08 |
| `k == n` | 5 | 5 | 1 | 1.51e-07 | 5.12e-08 |
| `k = 1` | 100 | 1 | 100 | 2.98e-08 | 2.32e-08 |
| `k = 2` | 100 | 2 | 99 | 1.11e-07 | 4.50e-08 |
| `k = 3` | 1,000 | 3 | 998 | 2.91e-07 | 3.23e-08 |
| `k = 16` | 1,024 | 16 | 1,009 | 2.97e-06 | 1.89e-07 |
| `k = 128` | 10,000 | 128 | 9,873 | 2.54e-05 | 5.52e-07 |
| `k = 257` | 65,537 | 257 | 65,281 | 4.48e-05 | 6.63e-07 |
| `k = 2047` | 100,000 | 2,047 | 97,954 | 3.13e-04 | 1.54e-06 |
| `k == n`, block | 260 | 260 | 1 | 5.83e-06 | 7.12e-07 |

The errors grow with `kernel_size` because the GPU accumulates in float32
while the reference accumulates in float64; they are consistent with float32
round-off.  The benchmark output was re-verified after timing (relative error
1.53e-06).

## Correctness — Numba CUDA

| Case | n | k | Outputs | Max abs error | Relative error |
|---|---:|---:|---:|---:|---:|
| challenge example 1 | 5 | 3 | 3 | 0.0 | 0.0 |
| challenge example 2 | 4 | 2 | 3 | 1.19e-07 | 2.59e-08 |
| single element | 1 | 1 | 1 | 4.16e-09 | 1.31e-08 |
| `k == n` | 5 | 5 | 1 | 3.64e-08 | 9.97e-08 |
| `k = 1` | 100 | 1 | 100 | 4.95e-08 | 3.89e-08 |
| `k = 2` | 100 | 2 | 99 | 6.07e-08 | 3.16e-08 |
| `k = 3` | 1,000 | 3 | 998 | 6.18e-07 | 5.35e-08 |
| `k = 16` | 1,024 | 16 | 1,009 | 1.57e-06 | 1.14e-07 |
| `k = 128` | 10,000 | 128 | 9,873 | 1.99e-05 | 4.96e-07 |
| `k = 257` | 65,537 | 257 | 65,281 | 2.35e-05 | 4.19e-07 |
| `k = 2047` | 100,000 | 2,047 | 97,954 | 1.75e-04 | 1.27e-06 |
| `k == n`, block | 260 | 260 | 1 | 2.43e-08 | 1.54e-09 |

## Performance — n = 1,500,000, k = 2,047

| Implementation | Median time | GFLOP/s |
|---|---:|---:|
| starter (shared tile, 1 output/thread, 2 shared loads/MAC) | 16.625 ms | 369 |
| **CUDA C++ final** | **3.34–3.39 ms** | **1,810–1,840** |
| Numba CUDA | 13.944 ms | 440 |

The final CUDA kernel is **≈5× faster than the starter** and reaches ~62% of
the P2000's fp32 peak (≈2.9 TFLOP/s).  Numba trails by 4.1× because it cannot
issue vector shared-memory loads: the C++ inner loop moves four taps and four
inputs per 16-byte load, while Numba must use scalar loads (one shared read
per MAC).

### CUDA variants benchmarked

| Variant | Time |
|---|---:|
| starter (1 output/thread, no register blocking) | 16.625 ms |
| 4 outputs/thread, 256 threads | 5.277 ms |
| 8 outputs/thread, 256 threads | 3.787 ms |
| 8 outputs/thread, 384 threads | 3.852 ms |
| 8 outputs/thread, 512 threads | 3.44–3.56 ms |
| **8 outputs/thread, 640 threads (final)** | **3.33–3.39 ms** |
| 8 outputs/thread, 704 threads | 3.39 ms |
| 8 outputs/thread, 768 threads | 3.41 ms |
| 16 outputs/thread, 256 threads | 3.92 ms |
| 16 outputs/thread, 384 threads | 3.95 ms |
| 4 outputs/thread, 512 threads | 6.73 ms |

(The 8-outputs/1024-threads configuration needs >48 KB of static shared
memory and cannot be launched.)  Removing the window-copy instructions
(`A0 = A1`) changes nothing — ptxas hides them — so the loop is bound by FMA
issue and shared-load throughput rather than register moves.

### Numba variants benchmarked

| Mapping | Time |
|---|---:|
| 1 output/thread (starter style) | 23.8 ms |
| 2 outputs/thread | 19.0 ms |
| 4 outputs/thread | 17.5 ms |
| 8 outputs/thread, no tap unroll | 21.6 ms |
| 8 outputs/thread, taps unrolled ×4 | 14.9 ms |
| 8 outputs/thread, taps unrolled ×8, 512 threads | 14.2 ms |
| **8 outputs/thread, taps unrolled ×8, 768 threads (final)** | **13.9 ms** |

## Build / run

CUDA C++ (the file exports `solve`, so compile it with a harness that
provides `main`):

```bash
nvcc -O3 -arch=sm_61 -o test convolution_1d.cu test_harness.cu
```

Numba:

```bash
python3 convolution_1d.py
```

The Python script runs the correctness cases and the performance case.  For
best performance, pass Numba device arrays; host arrays incur copies to and
from the device.
