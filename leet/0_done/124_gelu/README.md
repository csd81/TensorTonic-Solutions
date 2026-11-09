# GELU (exact, erf-based) — CUDA C++ and Numba Results

Challenge: elementwise exact GELU,
`out[i] = 0.5·x·(1 + erf(x/√2))` (the erf form, not the tanh approximation).
`1 ≤ N ≤ 100,000,000`; performance is measured at `N = 50,000,000`.

Files:

- `gelu.cu` — CUDA C++ implementation of `solve(X, Y, N)` (renamed from
  `starter.cu`).
- `gelu_naive.cu` — the naive baseline: one thread per element, scalar
  load/store and `erff`.
- `gelu.py` — Numba CUDA port with the same signature.

## The starter does not compile

Its launch reads `gelu_kernel<<<blocksGrid, threadsPerBlock>>>` while the
variable is named `blocksPerGrid` — an undeclared identifier.  Every timing
below for "starter" is from a copy with that one identifier fixed and nothing
else changed.

## Where the time goes at N = 50,000,000

The kernel reads 200 MB and writes 200 MB — 400 MB.  A pure `float4` copy of
that measures **4.85–4.88 ms (82 GB/s)**, and everything correct is on it:

| Variant | Time | GB/s |
|---|---:|---:|
| float4 copy floor | 4.85–4.88 ms | 82 |
| **shipped (float4, `erff`)** | **4.854–4.891 ms** | **82** |
| naive (scalar, `erff`) | 4.862–4.887 ms | 82 |
| starter (scalar, `erff`, typo fixed) | 4.879 ms | 82 |

At 400 MB the `erff` is completely hidden behind the memory traffic — in
contrast to GEGLU at 6 MB, where the same math was free but the launch was
not.  Vectorization is within noise (4.861 vs 4.879 in an ablation sweep).

Alternative formulations were measured too (`x * normcdff(x)` is the same
function): the float4 normcdff form ties at 4.861 ms, while the scalar
normcdff form is slower (5.16 ms).  `erff` is kept because the challenge asks
for the erf form and because `erff` is the more accurate of the two near
GELU's negative tail minimum.

## Implementation — CUDA C++

One `float4` per thread with 256-thread blocks, `erff`, and the scalar tail
(`N % 4`) handled in-kernel by the thread just past the last `float4`.  A
scalar kernel runs when either pointer is not 16-byte aligned.

### Naive baseline

`gelu_naive.cu` is the plainest correct form: one thread per element, scalar
load/store, `erff`.  It measures **4.86–4.89 ms** — the same as the shipped
kernel.  The naive mapping is already optimal for a 400 MB streaming kernel.

## Implementation — Numba

Scalar one-element-per-thread kernel using `numba.cuda.libdevice.erff` with
float32 constants (`math.erf` would compute in float64 in Numba and cost ~32×
more on this GPU — the trap documented in the GEGLU port).  It reaches
**4.951 ms (80.8 GB/s)**, 1.7% behind the C++ kernel, as expected once the
launch overhead disappears at this size.  Kernels use `cache=True`.

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

Checked against a float64 CPU reference with a per-element tolerance of
`1e-4 + 1e-5·|ref|` (absolute error dominates near GELU's zero-crossings and
its far negative tail):

| Case | Result |
|---|---|
| challenge example 1 (`[0,1,-1,2]` → `[0, 0.8413, -0.1587, 1.9545]`) | max_abs 2.4e-07 |
| challenge example 2 (`[0.5,-0.5,3,-3]` → `[0.3457, -0.1543, 2.9959, -0.0041]`) | max_abs 2.4e-07 |
| single zero | exact |
| extremes (-100, -50, -10, 10, 50, 100) | max_abs ≤ 4.6e-07 |
| tiny values (±1e-6, ±1e-30) | max_abs ≤ 3.8e-07 |
| sizes 1, 2, 3, 5, 7, 1023, 1024, 1025, 4099, 100,000, 50,000,000 | ≤ 4.6e-07 |
| misaligned `X+1`, `Y+1` (N = 1000) | ≤ 4.6e-07 |
| misaligned `X+2`, `Y+3` (N = 1001) | ≤ 4.6e-07 |
| benchmark shape, N = 50,000,000 | max_abs 4.6e-07 |

All 19 harness cases pass with zero elements outside tolerance.

## Correctness — Numba CUDA

All 15 cases pass (max_abs ≤ 4.5e-07 at the benchmark size).

## Performance — N = 50,000,000

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| starter (scalar, typo fixed) | 4.879 ms | 82 | 1.00× |
| `gelu_naive.cu` (scalar) | 4.862–4.887 ms | 82 | 1.00× |
| **`gelu.cu`** (float4) | **4.854–4.891 ms** | **82** | 1.0× |
| Numba CUDA | 4.951 ms | 80.8 | 1.02× slower |
| *float4 copy floor* | *4.85–4.88 ms* | *82* | *1.00×* |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test gelu.cu test_harness.cu
```

`THREADS` can be overridden with `-D` (the default is 256).

Numba:

```bash
python3 gelu.py
```

The script runs the correctness cases and the performance case.
