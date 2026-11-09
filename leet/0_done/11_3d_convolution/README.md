# 3D Convolution (valid) — CUDA C++ and Numba Results

Challenge: compute the "valid" 3D convolution

```
out[i][j][k] = sum_{d<KD, r<KR, c<KC} in[i+d][j+r][k+c] * k[d][r][c]
```

for volumes up to 256³ and kernels up to 5³.  The performance line specifies
`input_cols = 128`, `input_rows = 128`, `kernel_cols = 5`, `kernel_rows = 5`;
the depth is not listed, so the benchmark was measured on the natural
extension — a **128×128×128 input with a 5×5×5 kernel** (124³ outputs,
~238 M MACs) — plus other shapes for reference.

Files:

- `convolution_3d.cu` — CUDA C++ implementation of
  `solve(input, kernel, output, input_depth, input_rows, input_cols,
  kernel_depth, kernel_rows, kernel_cols)` (renamed from `starter.cu`).
- `convolution_3d.py` — Numba CUDA port with the same `solve` signature.

## Implementation — CUDA C++

The kernel is register tiled in all three dimensions:

| Parameter | Value |
|---|---:|
| Block | 16×16 threads (256) |
| Outputs per thread | 1 row × 4 columns × 4 depths |
| Outputs per block | 64×16×4 |
| Shared memory | kernel 800 B (padded) + input tile (dynamic) |

Per block:

1. the kernel is staged in shared memory with **each (d, r) row padded to a
   multiple of four floats and zero filled**, so four taps load as one
   aligned broadcast `float4`;
2. the input tile plus halo — `RD+KD-1` planes × `BH+KR-1` rows ×
   `BW+KC-1` columns — is staged in shared memory with a 16-byte-aligned,
   padded column stride (43.5 KB at the benchmark shape, so two blocks per
   SM).  Only the row padding needs zeroing; the staging loop writes every
   valid column, including the zero fill outside the input;
3. for each (kernel depth, kernel row), each thread reloads its 4-wide
   column window for each of its four output depths and slides along the
   columns four taps at a time: four `float4` loads (one per depth), one
   broadcast `float4` for four taps, and 64 FMAs.  A scalar tail handles
   `KC % 4` leftover taps.

Depth blocking (`RD = 4`) is what makes the tile worthwhile: one input plane
is reused by four output depths, cutting the halo traffic per output.

Kernels with any dimension > 5 are rejected (the challenge guarantees ≤ 5).
Note that the starter instead overflows its 125-float `__constant__` array
for kernels larger than 5³ — an out-of-spec input that the final kernel
handles by returning early.

## Implementation — Numba

Same mapping and loop structure with scalar window registers (Numba has no
float4 shared loads) and dynamic shared memory (`shape=0` + `sharedmem=`).

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, sm_61) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-08 |

Kernel times are medians of 20 runs (C++) / 20 runs (Numba) after a warm-up,
with inputs on the device, including `cudaDeviceSynchronize()`.  Correctness
is checked against a float64 CPU reference (full convolution for small
cases, sampled output positions for the benchmark size).

## Correctness — CUDA C++

| Case | Input | Kernel | Output | Relative error |
|---|---|---|---:|---:|
| challenge example 1 | 3×3×3 | 2×3×3 | 2×1×1 | 0.0 |
| challenge example 2 | 2×2×2 | 2×2×2 | 1×1×1 | 0.0 |
| single element | 1×1×1 | 1×1×1 | 1×1×1 | 2.36e-08 |
| `k = 1³` | 5×5×5 | 1×1×1 | 5×5×5 | 3.16e-08 |
| `k = 5³` | 9×9×9 | 5×5×5 | 5×5×5 | 4.36e-07 |
| mixed kernel | 8×8×8 | 3×5×2 | 6×4×7 | 1.47e-07 |
| odd volume | 17×19×23 | 5×5×5 | 13×15×19 | 3.45e-07 |
| `k == input` | 5×5×5 | 5×5×5 | 1×1×1 | 8.18e-08 |
| `k = 1×5×5` | 10×10×10 | 1×5×5 | 10×6×6 | 1.35e-07 |
| `k = 5×1×1` | 10×10×10 | 5×1×1 | 6×10×10 | 6.36e-08 |
| even kernel | 33×31×29 | 4×4×4 | 30×28×26 | 2.66e-07 |
| 64³ volume | 64×64×64 | 5×5×5 | 60×60×60 | 4.10e-07 |

The benchmark output was re-verified after timing (relative error 3.55e-07).

## Correctness — Numba CUDA

| Case | Input | Kernel | Relative error |
|---|---|---|---:|
| challenge example 1 | 3×3×3 | 2×3×3 | 0.0 |
| challenge example 2 | 2×2×2 | 2×2×2 | 0.0 |
| single element | 1×1×1 | 1×1×1 | 1.31e-08 |
| `k = 1³` | 5×5×5 | 1×1×1 | 4.28e-08 |
| `k = 5³` | 9×9×9 | 5×5×5 | 2.46e-07 |
| mixed kernel | 8×8×8 | 3×5×2 | 1.02e-07 |
| odd volume | 17×19×23 | 5×5×5 | 3.76e-07 |
| `k == input` | 5×5×5 | 5×5×5 | 4.77e-07 |
| `k = 1×5×5` | 10×10×10 | 1×5×5 | 9.50e-08 |
| `k = 5×1×1` | 10×10×10 | 5×1×1 | 6.52e-08 |
| even kernel | 33×31×29 | 4×4×4 | 2.54e-07 |
| 64³ volume | 64×64×64 | 5×5×5 | 5.92e-07 |

## Performance

| Implementation | Shape | Median time | GFLOP/s |
|---|---|---:|---:|
| starter | 128³, k=5³ | 4.000 ms | 119 |
| **CUDA C++ final** | **128³, k=5³** | **1.219 ms** | **391** |
| Numba CUDA | 128³, k=5³ | 2.470 ms | 193 |

The final CUDA kernel is **3.3× faster than the starter** at the benchmark
shape.  Other shapes with the final kernel:

| Shape | Median time | GFLOP/s |
|---|---:|---:|
| 128³, k=3³ | 0.621 ms | 174 |
| 256³, k=5³ | 10.417 ms | 384 |
| 1×128×128, k=1×5×5 | 0.020 ms | 39 |

For the tiny `1×128×128` case the starter is marginally faster (0.017 ms)
because it has no shared-memory staging; the tiled kernel wins everywhere
the volume has real depth.

Ablation at 128³, k=5³ (1.22 ms total): input staging ≈ 0.39 ms, compute
loop ≈ 0.56 ms, masked stores + launch ≈ 0.27 ms.

### Tile configurations benchmarked (128³, k=5³)

| Block / tiling | Time |
|---|---:|
| starter (1 output/thread, constant memory) | 4.000 ms |
| 16×8 threads, RN=4, RD=4 | 1.353 ms |
| **16×16 threads, RN=4, RD=4 (final)** | **1.219 ms** |
| 16×16 threads, RD=3 | 1.380 ms |
| 16×12 threads, RD=4 | 1.405 ms |
| 32×8 threads, RN=4, RD=2 | 1.607 ms |
| 16×8 threads, RD=8 | 1.596 ms |
| 8×8 threads, RD=8 | 1.627 ms |
| 16×16 threads, RD=2 | 1.824 ms |
| 16×16 threads, RD=6 | > 48 KB dynamic shared, cannot launch |
| 16×16, RD=4, `float4`-vectorised staging | 1.447 ms |

Findings:

- Depth blocking matters: `RD=4` amortises the `KD-1` halo planes across four
  output depths (`RD=2` is 50% slower), while `RD=8` needs too much shared
  memory at useful block sizes.
- The shared-memory footprint (43.5 KB) allows two blocks per SM; larger
  tiles that would reduce halo traffic drop occupancy further and lose.
- Vectorising the staging loads to `float4` made it *slower* (1.45 vs
  1.22 ms); the scalar loader with its simple coalesced pattern wins here.

## Build / run

CUDA C++ (the file exports `solve`, so compile it with a harness that
provides `main`):

```bash
nvcc -O3 -arch=sm_61 -o test convolution_3d.cu test_harness.cu
```

Numba:

```bash
python3 convolution_3d.py
```

The Python script runs the correctness cases and the performance case.  For
best performance, pass Numba device arrays; host arrays incur copies to and
from the device.
