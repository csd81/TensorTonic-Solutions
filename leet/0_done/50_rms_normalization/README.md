# RMS Normalization — CUDA C++ and Numba Results

Challenge: RMS normalization of a 1D tensor,

```
rms = sqrt(mean(x_i^2) + eps)
y_i = gamma * x_i / rms + beta
```

with `eps = 1e-5`, `1 ≤ N ≤ 100,000`; performance is measured at
`N = 100,000` (400 KB of input).

Files:

- `rms_normalization.cu` — CUDA C++ implementation of
  `solve(input, gamma, beta, output, N, eps)` (renamed from `starter.cu`).
- `rms_normalization.py` — Numba CUDA port with the same `solve` signature.

## Implementation — CUDA C++

At this size the problem is **launch- and latency-bound**, not
bandwidth-bound: the whole working set is 1.2 MB of traffic, while each
kernel launch costs several microseconds.  The final implementation therefore
minimises launches and allocations:

- **one cooperative kernel**: sum of squares (float4 loads, one thread per
  four elements) → `cg::this_grid().sync()` → scale-and-shift pass.  The grid
  is capped to the co-resident block capacity reported by
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, as a cooperative launch
  requires;
- a **cached 4-byte device scratch** replaces the starter's per-call
  `cudaMalloc` / `cudaFree` (the free forces a device synchronisation and
  costs more than the kernels themselves);
- the reduction is one thread per four elements with `float4` loads, so only
  a few dozen `atomicAdd`s reach the counter (the starter launched 391
  blocks, i.e. 391 atomics, for 100,000 elements);
- a two-kernel fallback is kept for devices without cooperative launch, and
  unaligned pointers fall back to scalar-load kernels.

The `N % 4` tail is included in the sum of squares and in the output (a first
version missed it in the reduction, which is only visible for N not divisible
by 4).

## Implementation — Numba

Numba has no grid-wide barrier, so the port uses two kernels: one writes the
per-block sums of squares, and the second — each block redundantly — adds
those few dozen values, computes the scale, and applies it.  That avoids both
the atomics and a separate zeroing launch; scalar loads are used because
Numba cannot issue `float4` loads.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, sm_61, **6 SMs**) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Kernel times are medians of 100 runs after a warm-up, with the input already
on the device, including `cudaDeviceSynchronize()`.

## Correctness — CUDA C++

| Case | N | gamma | beta | Max abs error | Relative error |
|---|---:|---:|---:|---:|---:|
| challenge example 1 | 4 | 1.0 | 0.0 | 5.61e-08 | 3.84e-08 |
| challenge example 2 | 3 | 1.0 | 0.0 | 4.31e-08 | 3.11e-08 |
| single element | 1 | 1.0 | 0.0 | 3.84e-08 | 3.84e-08 |
| single zero | 1 | 1.0 | 0.0 | 0.0 | 0.0 |
| odd small | 31 | 1.0 | 0.0 | 7.57e-08 | 4.95e-08 |
| one block | 1,024 | 2.5 | −1.5 | 3.02e-07 | 5.23e-08 |
| block + 1 | 1,025 | 0.1 | 10.0 | 4.81e-07 | 4.73e-08 |
| near benchmark | 99,999 | 10.0 | −10.0 | 1.80e-06 | 6.58e-08 |
| benchmark size | 100,000 | 1.0 | 0.0 | 1.12e-07 | 6.49e-08 |
| all zeros | 4,096 | 3.0 | 2.0 | 0.0 | 0.0 |

The benchmark output was re-verified after timing (relative error 7.0e-08).

## Correctness — Numba CUDA

All ten cases above pass as well; relative errors range from 0 (all-zero
input) to 2.0e-07 at the benchmark size.

## Performance — N = 100,000

| Implementation | Median time | Effective bandwidth |
|---|---:|---:|
| starter (two kernels, `cudaMalloc`/`cudaFree` per call) | 42.0 µs | 28.6 GB/s |
| **CUDA C++ final (single cooperative kernel)** | **15.5–15.9 µs** | **~76 GB/s** |
| Numba CUDA | 132.2 µs | 9.1 GB/s |

The final CUDA kernel is **2.7× faster than the starter**.  The remaining
time is roughly half memory (1.2 MB at ~76 GB/s) and half the launch and
grid-sync latency — there is little left to win at this size.

The Numba port is 8.5× slower purely because of launch overhead: each Numba
kernel dispatch costs tens of microseconds on this setup, and the actual
kernel work is only ~15 µs.  A three-kernel variant measured 138 µs, so the
two-kernel structure was kept.

### Notes

- A cooperative launch silently fails if the grid exceeds the number of
  co-resident blocks — the first version asked for 64 blocks on a 6-SM GPU
  (48 co-resident), the launch was rejected and the output stayed zero.  The
  final code asks the occupancy API and checks the launch result, falling
  back to the two-kernel path on failure.
- The starter's `cudaFree` at the end of `solve` happens to serialise the
  device, so its result is correct, but the allocation dominates its runtime
  at this problem size.

## Build / run

CUDA C++ (the file exports `solve`, so compile it with a harness that
provides `main`):

```bash
nvcc -O3 -arch=sm_61 -o test rms_normalization.cu test_harness.cu
```

Numba:

```bash
python3 rms_normalization.py
```

The Python script runs the correctness cases and the performance case.  For
best performance, pass Numba device arrays; host arrays incur copies to and
from the device.

## Naive baseline

`rms_normalization_naive.cu` is the same computation written as plainly as possible:
two plain kernels: a serial per-chunk sum-of-squares reduction, then one thread per element re-summing the partials and normalising — no shared-memory tiling, no vector types, no warp intrinsics and no
fast-math.  It passes the same correctness cases as the optimized kernel
(checked against a float64 CPU reference in the shared harness).

| Implementation | Time | vs naive |
|---|---:|---:|
| `rms_normalization_naive.cu` | 0.4983 ms | 1.00x |
| `rms_normalization.cu` (optimized) | 0.0149 ms | 33.5x |

At N = 100,000 the problem is launch- and latency-bound; the optimized kernel does it in one cooperative launch.
