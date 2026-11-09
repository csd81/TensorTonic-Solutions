# 5-Point Jacobi Stencil — CUDA C++ and Numba Results

Challenge: one Jacobi iteration on an `rows × cols` float grid — interior
cells become `0.25 × (top + bottom + left + right)` of the input, boundary
cells are copied through.  `rows, cols ≤ 16,384`; performance is measured at
`8192 × 8192`.

Files:

- `jacobi_stencil_2d.cu` — CUDA C++ implementation of
  `solve(input, output, rows, cols)` (renamed from `starter.cu`).
- `jacobi_stencil_2d_naive.cu` — the naive baseline: one thread per cell,
  four scalar neighbour loads.
- `jacobi_stencil_2d.py` — Numba CUDA port with the same signature.

## Where the time goes at 8192 × 8192

The grid is 67 M cells = 268 MB read + 268 MB written = **537 MB**.  A plain
float4 copy of that measures **6.68–7.22 ms (74–80 GB/s)**, and the stencil
runs at essentially the same rate because neighbour reuse is served by the
cache.

| Variant | Time | GB/s |
|---|---:|---:|
| starter / naive (scalar, 32×32 blocks) | 7.19–7.74 ms | 69–75 |
| scalar, 32×8 blocks | 7.37 ms | 72.9 |
| **shipped (float4, 4 cells/thread, 64×4 blocks)** | **6.73–7.03 ms** | **76–80** |
| float4, 32×2 blocks (one session's sweep) | 6.54 ms | 82.1 |
| float4, 64×2 blocks | 6.56 ms | 81.9 |
| float4, 32×8 blocks | 7.31 ms | 73.4 |

The gain over the starter is ~8–17% depending on the session (6.54 vs 7.74 in
the sweep session; 6.73 vs 7.19 in the interleaved harness run) — the
stencil is near the DRAM wall either way.

## Implementation — CUDA C++

Four adjacent columns per thread:

- Top, bottom and centre rows are loaded as `float4` (three vector loads for
  four cells); the two out-of-group neighbours (left of the first cell, right
  of the last) are scalar loads.
- Left/right neighbours are formed in registers from the centre vector
  (`left = [l0, mid.x, mid.y, mid.z]`, `right = [mid.y, mid.z, mid.w, r3]`).
- Boundary rows and the two boundary-column groups take the scalar path
  (4 cells out of every 4·cols).
- Small blocks win: 64×4 (256 threads) is the shipped shape; 32×2, 64×2 and
  64×4 all measure ~6.5 ms, while 32×8 and 16×16 are ~7.3–7.5.

The vector path requires `cols % 4 == 0` and 16-byte-aligned pointers; a
scalar one-cell-per-thread kernel covers everything else.

### Naive baseline

`jacobi_stencil_2d_naive.cu` is the plainest correct form — the starter's
scalar shape with 32×32 blocks and four scalar neighbour loads.  It measures
**7.19–7.64 ms**, ~8% slower than the shipped kernel.

## Implementation — Numba

Scalar one-cell-per-thread with 32×8 2-D blocks (Numba has no float4 type).
It reaches **7.256 ms (74.0 GB/s)**, in line with the C++ scalar form — at
537 MB the launch overhead is irrelevant.  Kernels use `cache=True`.

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

Checked against a float64 CPU reference with a relative tolerance of 1e-4:

| Case | Result |
|---|---|
| challenge example (4×4, interior values unchanged) | exact |
| 1×1, 2×2 (all boundary) | exact |
| 3×5, 5×3, 17×33, 64×64, 100×37 | max_abs ≤ 7.7e-06 |
| benchmark shape, 8192×8192 | max_abs ≤ 1e-05 |

All 9 harness cases pass.

## Correctness — Numba CUDA

All 8 cases pass with the same tolerance.

## Performance — 8192 × 8192

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| `jacobi_stencil_2d_naive.cu` (scalar, 32×32) | 7.19–7.64 ms | 70–75 | 1.07–1.09× slower |
| starter (same shape) | 7.2–7.7 ms | 69–75 | 1.07–1.10× slower |
| **`jacobi_stencil_2d.cu`** (float4, 4 cells/thread) | **6.73–7.03 ms** | **76–80** | 1.0× |
| Numba CUDA (scalar) | 7.256 ms | 74.0 | 1.03–1.08× slower |
| *float4 copy floor* | *6.68–7.22 ms* | *74–80* | *1.00×* |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test jacobi_stencil_2d.cu test_harness.cu
```

Numba:

```bash
python3 jacobi_stencil_2d.py
```

The script runs the correctness cases and the performance case.
