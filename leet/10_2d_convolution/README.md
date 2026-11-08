# 2D Convolution (valid) — CUDA C++ and Numba Results

Challenge: compute the "valid" 2D convolution

```
out[i][j] = sum_{m < kernel_rows, n < kernel_cols} in[i+m][j+n] * k[m][n]
```

for inputs up to 3072×3072 and kernels up to 31×31.  Performance is measured
with a 3072×3072 input and a 15×15 kernel (3058×3058 outputs, ~2.10 G MACs).

Files:

- `convolution_2d.cu` — CUDA C++ implementation of
  `solve(input, kernel, output, input_rows, input_cols, kernel_rows,
  kernel_cols)` (renamed from `starter.cu`).
- `convolution_2d.py` — Numba CUDA port with the same `solve` signature.

## Implementation — CUDA C++

The workload is compute bound, so the kernel is register tiled and tuned to
issue many FMAs per shared-memory access:

| Parameter | Value |
|---|---:|
| Block | 32×8 threads (256) |
| Outputs per thread | 2 rows × 4 columns |
| Outputs per block | 128×16 |
| Shared memory | kernel 992 floats (padded) + input tile (dynamic) |

Per block:

1. the kernel is staged in shared memory with **each row padded to a
   multiple of four floats and zero filled**, so four taps load as one
   aligned broadcast `float4`;
2. the input tile plus halo (`128+KC-1` by `16+KR-1`) is staged in shared
   memory with a 16-byte-aligned, padded row stride; out-of-range elements
   are zeroed;
3. per kernel row `m`, each thread keeps its 2×4 input window in registers
   (two `float4`s per row) and walks the kernel columns four at a time:
   two `float4` loads bring the next four input columns, one broadcast
   `float4` brings four taps, and 32 FMAs are issued; the window then
   advances by four columns.

This reduces the per-MAC cost from ~3 instructions (starter: one shared
load + one constant load + one FMA per MAC, with a single serial
accumulator) to ~1.3, and removes the starter's `cudaMemcpyToSymbol` and
constant-memory dependency.

## Implementation — Numba

Numba has no vector shared-memory loads, so the same 2×4 tiling is written
with scalar window registers: per four taps it loads 8 new input values, 4
taps, and issues 32 FMAs.  Dynamic shared memory (`shape=0` +
`sharedmem=`) sizes the tile to the actual kernel.

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
is checked against a float64 CPU reference (full convolution for small
cases, sampled output positions for the benchmark size).

## Correctness — CUDA C++

| Case | Input | Kernel | Output | Max abs error | Relative error |
|---|---|---|---:|---:|---:|
| challenge example 1 | 3×3 | 2×2 | 2×2 | 0.0 | 0.0 |
| challenge example 2 | 4×4 | 1×3 | 4×2 | 0.0 | 0.0 |
| single element | 1×1 | 1×1 | 1×1 | 1.49e-08 | 2.36e-08 |
| `k = 1×1` | 5×5 | 1×1 | 5×5 | 1.11e-07 | 3.68e-08 |
| `k = 1×3` | 100×100 | 1×3 | 100×98 | 2.90e-07 | 6.11e-08 |
| `k = 3×1` | 100×100 | 3×1 | 98×100 | 5.31e-07 | 5.59e-08 |
| `k = 7×5` | 100×100 | 7×5 | 94×96 | 3.67e-06 | 2.29e-07 |
| odd sizes | 257×129 | 15×15 | 243×115 | 3.84e-05 | 5.94e-07 |
| `k = 31×31` | 64×64 | 31×31 | 34×34 | 9.88e-05 | 7.90e-07 |
| `k = 1×31` | 33×31 | 1×31 | 33×1 | 9.60e-07 | 1.34e-07 |
| `k == input` | 17×19 | 17×19 | 1×1 | 4.09e-06 | 1.30e-07 |
| 2×2 kernel | 300×200 | 2×2 | 299×199 | 7.94e-07 | 1.07e-07 |

Errors are float32 round-off relative to the float64 reference and grow with
the number of taps.  The benchmark output was re-verified after timing
(relative error 4.38e-07).

## Correctness — Numba CUDA

| Case | Input | Kernel | Output | Max abs error | Relative error |
|---|---|---|---:|---:|---:|
| challenge example 1 | 3×3 | 2×2 | 2×2 | 0.0 | 0.0 |
| challenge example 2 | 4×4 | 1×3 | 4×2 | 0.0 | 0.0 |
| single element | 1×1 | 1×1 | 1×1 | 4.16e-09 | 1.31e-08 |
| `k = 1×1` | 5×5 | 1×1 | 5×5 | 1.40e-08 | 1.97e-08 |
| `k = 1×3` | 100×100 | 1×3 | 100×98 | 2.74e-07 | 7.11e-08 |
| `k = 3×1` | 100×100 | 3×1 | 98×100 | 5.22e-07 | 6.66e-08 |
| `k = 7×5` | 100×100 | 7×5 | 94×96 | 7.15e-06 | 2.65e-07 |
| odd sizes | 257×129 | 15×15 | 243×115 | 3.95e-05 | 5.94e-07 |
| `k = 31×31` | 64×64 | 31×31 | 34×34 | 1.06e-04 | 9.26e-07 |
| `k = 1×31` | 33×31 | 1×31 | 33×1 | 9.43e-07 | 6.91e-08 |
| `k == input` | 17×19 | 17×19 | 1×1 | 1.43e-05 | 5.19e-07 |
| 2×2 kernel | 300×200 | 2×2 | 299×199 | 7.33e-07 | 7.62e-08 |

## Performance — 3072×3072 input, 15×15 kernel

| Implementation | Median time | GFLOP/s |
|---|---:|---:|
| starter (1 output/thread, constant-memory taps) | 18.101 ms | 233 |
| **CUDA C++ final** | **3.48–3.55 ms** | **1,190–1,210** |
| Numba CUDA | 12.180 ms | 346 |

The final CUDA kernel is **≈5.15× faster than the starter**.  Numba trails by
3.5× because it cannot issue vector shared-memory loads (8 scalar loads per
four taps instead of two `float4`s).

Ablation of the final kernel at the benchmark size: tile loading + store
traffic alone is ~1.29 ms (essentially DRAM bandwidth for ~100 MB moved),
the compute loop is ~2.5 ms (FMA-issue floor ≈1.2 ms, instruction floor
≈2.1 ms).

### CUDA variants benchmarked

| Variant | Time |
|---|---:|
| starter (scalar, 1 output/thread, constant memory) | 18.101 ms |
| 2×4 register tile, scalar tap loads, old tile loader | 8.343 ms |
| + padded kernel rows, `float4` taps, 2D tile loader | 3.794 ms |
| same, 16×16 threads, 2×4 tile | 3.800 ms |
| same, 16×16 threads, 4×4 tile | 3.560 ms |
| **same, 32×8 threads, 2×4 tile (final)** | **3.517 ms** |
| same, 32×8 threads, 4×4 tile | 3.527 ms |
| same, 8×32 threads, 2×4 tile | 3.781 ms |
| same, 32×16 threads, 2×4 tile | 4.305 ms |
| same, 32×16 threads, 4×4 tile | 4.026 ms |
| compile-time-specialised 15×15 kernel (earlier structure) | 4.777 ms |

Two findings drove the final kernel:

- **Padding the kernel rows to a multiple of four floats** turns four scalar
  broadcast loads into one aligned `float4` broadcast and lets the tap loop
  run without a scalar remainder.
- The old row-by-row tile loader left most threads idle (only `tile_w`
  threads active per row); the 2D loader cut the load phase from ~1.8 ms to
  ~0.44 ms.

## Build / run

CUDA C++ (the file exports `solve`, so compile it with a harness that
provides `main`):

```bash
nvcc -O3 -arch=sm_61 -o test convolution_2d.cu test_harness.cu
```

Numba:

```bash
python3 convolution_2d.py
```

The Python script runs the correctness cases and the performance case.  For
best performance, pass Numba device arrays; host arrays incur copies to and
from the device.
