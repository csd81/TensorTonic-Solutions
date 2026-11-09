# Sigmoid — CUDA C++ and Numba Results

Challenge: elementwise sigmoid `out[i] = 1 / (1 + exp(-x))` on a float32
vector.  `1 ≤ N ≤ 100,000,000`; performance is measured at `N = 50,000,000`.

Files:

- `sigmoid.cu` — CUDA C++ implementation of `solve(X, Y, N)` (renamed from
  `starter.cu`).
- `sigmoid_naive.cu` — the naive baseline: one thread per element, scalar
  loads, `expf` and IEEE division.
- `sigmoid.py` — Numba CUDA port with the same signature.

## Where the time goes at N = 50,000,000

The kernel reads 200 MB and writes 200 MB — **400 MB**.  A pure `float4` copy
of that measures **4.84–5.00 ms (80–83 GB/s)**, and the implementations line
up against it:

| Variant | Time | GB/s |
|---|---:|---:|
| float4 copy floor | 4.84–5.00 ms | 80–83 |
| **shipped (float4, `__expf` + `__fdividef`)** | **4.84–4.92 ms** | **81–83** |
| naive (scalar, `expf` + IEEE division) | 5.06–5.14 ms | 78–79 |
| starter (same shape) | 5.07–5.17 ms | 77–79 |

Unlike ReLU (all zeros and selects), the transcendental and the division
are just visible at this size: the full-precision scalar form sits ~4% above
the copy floor, and `__expf` + `__fdividef` plus float4 closes the gap.

## Implementation — CUDA C++

One `float4` per thread with 256-thread blocks, `__expf` + `__fdividef`, and
the scalar tail (`N % 4`) handled in-kernel by the thread just past the last
`float4`.  A scalar kernel runs when either pointer is not 16-byte aligned.
The fast intrinsics cost ~1e-6 relative error (documented below), far inside
any reasonable tolerance for a value in `(0, 1)`.

### Naive baseline

`sigmoid_naive.cu` is the plainest correct form: one thread per element,
`expf` and IEEE division.  It measures **5.06–5.14 ms**, 1.05× slower than
the shipped kernel — the gap is exactly the arithmetic the fast intrinsics
remove.

## Implementation — Numba

Scalar one-element-per-thread kernel.  Getting it to match C++ took two
Numba-specific fixes, both worth documenting:

1. `math.exp` computes in **float64** in Numba (`inspect_types` shows a
   float64 result), and this GPU's FP64 rate is 1/32 of FP32: the first port
   measured **12.48 ms**.
2. Switching to `libdevice.expf` (float32) brought it to **5.47 ms**; using
   `libdevice.fast_expf` + `libdevice.fast_fdividef` (the `__expf` /
   `__fdividef` equivalents) reached **4.88 ms (81.9 GB/s)** — the same as
   the C++ kernel, since at 400 MB the launch overhead disappears and only
   the arithmetic quality matters.

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

Times are medians after a warm-up, including `cudaDeviceSynchronize()`.

## Correctness — CUDA C++

Checked against a float64 CPU reference; the harness requires every element
within 1e-5 absolute:

| Case | Result |
|---|---|
| challenge example 1 (`[0,1,-1,2]`) | max_abs 1.9e-08 |
| challenge example 2 (`[0.5,-0.5,3,-3]`) | max_abs 2.1e-08 |
| single zero | exact |
| extremes (-100, -50, -10, 10, 50, 100) | max_abs 2.1e-08 |
| tiny values (±1e-6, ±1e-30) | max_abs 4.8e-08 |
| sizes 1, 2, 3, 5, 7, 1023, 1024, 1025, 4099, 100,000, 50,000,000 | ≤ 1.2e-07 |
| misaligned `X+1`, `Y+1` (N = 1000) | ≤ 1.2e-07 |
| misaligned `X+2`, `Y+3` (N = 1001) | ≤ 1.2e-07 |
| benchmark shape, N = 50,000,000 | max_abs 1.2e-07 |

All 19 harness cases pass with zero elements outside 1e-5.

## Correctness — Numba CUDA

All 15 cases pass with max_abs ≤ 1.2e-07 (the fast intrinsics give the same
accuracy class as the C++ kernel).

## Performance — N = 50,000,000

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| `sigmoid_naive.cu` (scalar, `expf` + division) | 5.06 ms | 79 | 1.05× slower |
| starter (same shape) | 5.07 ms | 79 | 1.05× slower |
| **`sigmoid.cu`** (float4, `__expf` + `__fdividef`) | **4.88 ms** | **82** | 1.0× |
| Numba CUDA (fast intrinsics) | 4.88 ms | 82 | 1.00× |
| *float4 copy floor* | *4.84–5.00 ms* | *80–83* | *1.00×* |

Everything correct is on the copy floor within noise; the challenge is
bandwidth-bound once the arithmetic is made cheap.

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test sigmoid.cu test_harness.cu
```

`THREADS` can be overridden with `-D` (the default is 256).

Numba:

```bash
python3 sigmoid.py
```

The script runs the correctness cases and the performance case.
