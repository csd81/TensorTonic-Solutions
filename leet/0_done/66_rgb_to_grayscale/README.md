# RGB to Grayscale — CUDA C++ and Numba Results

Challenge: convert an RGB image (interleaved `R,G,B,R,G,B,…` triples, all
float32) to grayscale with
`out[p] = 0.299·R + 0.587·G + 0.114·B`.  `1 ≤ width, height ≤ 4096`,
`width·height ≤ 4,194,304`; performance is measured at `2048 × 2048`.

Files:

- `rgb_to_grayscale.cu` — CUDA C++ implementation of
  `solve(input, output, width, height)` (renamed from `starter.cu`).
- `rgb_to_grayscale_naive.cu` — the naive baseline: one thread per pixel,
  three scalar loads.
- `rgb_to_grayscale.py` — Numba CUDA port with the same signature.

## The starter is broken

Its pixel index is

```c
int pixel_idx = blockIdx.x * blockIdx.y * blockDim.x + threadIdx.x;
```

and the launch is 1-D, so `blockIdx.y == 0` for every block.  Every block
therefore recomputes **pixels 0–255**, and the rest of the output is never
written.  Correctness fails on every case (max_abs = ∞, `rel = 1.0`), and
its timing is meaningless too — it reads and writes a tiny fraction of the
traffic but reports 0.44 ms (152 "GB/s").  The fix is the plain global 1-D
index.

## Where the time goes at 2048 × 2048

4,194,304 pixels: 50.3 MB read + 16.8 MB written = **67 MB**.  The same
traffic with the least possible math (`out = r + g + b`, same access pattern)
measures **0.800 ms (83.9 GB/s)**, and the correct implementations are at
that floor:

| Variant | Time | GB/s |
|---|---:|---:|
| sum floor (same traffic, minimal math) | 0.799–0.802 ms | 83.7–84.0 |
| naive (scalar, 1 pixel/thread) | 0.796–0.805 ms | 83.4–84.4 |
| **shipped (scalar + `__ldg`, 256 thr)** | **0.803–0.806 ms** | **83.3–83.6** |
| float4, 4 pixels/thread (3 float4 loads/group) | 0.807–0.810 ms | 82.8–83.2 |

The vectorized form — 12 consecutive floats per group, loaded as three
16-byte-aligned `float4`s and stored as one `float4` — is 1% *slower* than
scalar: DRAM is the limit, and the wider accesses only add register pressure.
So the scalar mapping ships.

## Implementation — CUDA C++

One thread per pixel, three `__ldg` loads and one store, 256-thread blocks,
one launch.  No vector types, no alignment checks — scalar accesses are valid
for any pointer and any `width·height`.

### Naive baseline

`rgb_to_grayscale_naive.cu` is the plainest correct form: one thread per
pixel, three scalar loads and the weighted sum.  It measures **0.80 ms** —
the same as the shipped kernel within noise.  The meaningful difference in
this challenge is the *correct* index, not the mapping.

## Implementation — Numba

Same scalar mapping.  It reaches **0.842 ms (79.7 GB/s)**, 4.9% behind the
C++ kernel; at 67 MB the per-launch overhead is negligible.  Kernels use
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
C++ variants were compiled against the same harness and run sequentially.

## Correctness — CUDA C++

Checked against a float64 CPU reference:

| Case | Result |
|---|---|
| challenge example 1 (2×2, `[76.245, 149.685, 29.07, 128.0]`) | max_abs 4.9e-06 |
| challenge example 2 (1×1, `[140.75]`) | max_abs 0 |
| 1×1, 1×5, 5×1, 3×7, 64×64, 100×100 | rel ≤ 1e-07 |
| 1000×1, 1122×1 (misaligned pointers) | rel ≤ 1e-07 |
| benchmark shape, 2048×2048 | max_abs 2.5e-05 |

The non-zero errors are float32 rounding of the weighted sum; all 12 harness
cases pass (13 with the benchmark-shape output check).

## Correctness — Numba CUDA

All 7 cases pass against the same kind of float64 reference (max_abs ≤ 2.6e-05).

## Performance — 2048 × 2048

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| starter (broken index, wrong output) | 0.44 ms | — | invalid |
| `rgb_to_grayscale_naive.cu` (scalar) | 0.800 ms | 83.9 | 1.00× |
| **`rgb_to_grayscale.cu`** (scalar + `__ldg`) | **0.803 ms** | **83.6** | 1.0× |
| Numba CUDA | 0.842 ms | 79.7 | 1.05× slower |
| *sum floor (same traffic)* | *0.800 ms* | *83.9* | *1.00×* |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test rgb_to_grayscale.cu test_harness.cu
```

`THREADS` can be overridden with `-D` (the default is 256).

Numba:

```bash
python3 rgb_to_grayscale.py
```

The script runs the correctness cases and the performance case.
