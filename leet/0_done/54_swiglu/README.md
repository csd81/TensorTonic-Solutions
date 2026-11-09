# SWiGLU — CUDA C++ and Numba Results

Challenge: SWiGLU on a float32 vector — split `x` into halves
`x1 = x[0..N/2)`, `x2 = x[N/2..N)` and compute
`out[i] = SiLU(x1[i]) * x2[i]` (`N/2` outputs).  `1 ≤ N ≤ 100,000`, `N` even;
performance is measured at `N = 100,000`.

Files:

- `swiglu.cu` — CUDA C++ implementation of `solve(input, output, N)` (renamed
  from `starter.cu`).
- `swiglu_naive.cu` — the naive baseline: one thread per output, scalar
  loads, `expf` and IEEE division.
- `swiglu.py` — Numba CUDA port with the same signature.

## Where the time goes at N = 100,000

The kernel reads 400 KB (both halves) and writes 200 KB, so it is
**launch-bound** at 600 KB of traffic:

| Baseline | Time |
|---|---:|
| empty kernel launch + `cudaDeviceSynchronize()` | 6.7 µs |
| same-traffic kernel (`out = x1 + x2`, float4) | 8.0 µs |
| this SWiGLU kernel (float4, `__expf` + `__fdividef`) | 8.0–8.5 µs |
| starter (scalar, `__expf` + IEEE division) | 12.3 µs |
| naive (scalar, `expf` + IEEE division) | 14.0 µs |

Ablation over the same 200-launch medians:

| Variant | Time | max abs err |
|---|---:|---:|
| scalar, `expf` + IEEE division (naive) | 11.9 µs | 4.9e-04 |
| scalar, `__expf` + `__fdividef` | 8.9 µs | 4.9e-04 |
| float4, `expf` + IEEE division | 11.2 µs | 4.9e-04 |
| **float4, `__expf` + `__fdividef`** | **8.0 µs** | 4.9e-04 |
| float4 fast, 128 / 512 threads | 8.0–8.1 µs | 4.9e-04 |

Two independent costs, both worth removing: the IEEE division and the
full-precision `expf` (~3 µs together; the starter already uses `__expf`,
hence its smaller gap to the naive form), and scalar accesses (~0.9 µs on
top).  With both fixed the kernel sits on the traffic floor.  The fast-math
error is invisible here: the worst absolute error (4.9e-04 on products up to
~10,000) is exactly the same for every variant, i.e. the float32 rounding of
the result, not the intrinsic.

## Implementation — CUDA C++

One `float4` per thread with 256-thread blocks: four outputs per thread, two
vector loads (the two halves are separate streams `halfN` apart) and one
vector store.  The vector path requires `halfN % 4 == 0` so the second half
starts 16-byte aligned, plus 16-byte-aligned pointers; otherwise a scalar
kernel runs.  Allocation: 50,000 outputs → 12,500 float4s → 49 blocks.

### Naive baseline

`swiglu_naive.cu` is the plainest correct form: one thread per output, two
scalar loads, `expf` and IEEE division.  It measures **14.0 µs**, 1.65×
slower than the shipped kernel.  The starter (12.3 µs) differs from it only
in already using `__expf`, which is a good illustration of where the time
goes: the transcendental/division choice is worth more than vectorization at
this size.

## Implementation — Numba

Scalar one-element-per-thread kernel with `math.exp` and IEEE division
(no float4 and no fast intrinsics available).  It measures **80.6 µs** —
9.5× the C++ kernel, almost entirely Python-side dispatch: at 600 KB the
~20 µs launch overhead plus marshalling dominates.  Kernels use
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
C++ variants were compiled against the same harness and run sequentially on
an idle GPU.  At this size run-to-run jitter is ~±0.5 µs.

## Correctness — CUDA C++

Checked against a float64 CPU reference over inputs in `[-100, 100]`:

| Case | Result |
|---|---|
| challenge example 1 (`[1,2,3,4]` → `[2.1931758, 7.0463767]`) | max_abs 5e-04 |
| challenge example 2 (`[0.5,1.0]` → `[0.31122968]`) | max_abs 5e-04 |
| all zeros / all equal / extremes (±100) | max_abs ≤ 4.9e-04 |
| even sizes 2, 4, 6, 10, 1024, 1026, 4098, 100,000 | rel ≤ 9.9e-07 |
| misaligned `input+1`, `output+1` (N = 1000) | rel ≤ 9.9e-07 |
| misaligned `input+2`, `output+3` (N = 1002) | rel ≤ 9.9e-07 |
| benchmark shape, N = 100,000 | max_abs 4.9e-04 |

The absolute errors are ~1e-6 relative to outputs of magnitude up to ~10⁴;
all 16 harness cases pass with max_rel < 1e-5.

## Correctness — Numba CUDA

All 13 cases pass with max_rel ≤ 2.2e-07 and max_abs ≤ 4.9e-04.

## Performance — N = 100,000

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| `swiglu_naive.cu` (scalar, `expf` + division) | 14.0 µs | 42.9 | 1.65× slower |
| starter (scalar, `__expf` + division) | 12.3 µs | 48.8 | 1.45× slower |
| **`swiglu.cu`** (float4, `__expf` + `__fdividef`) | **8.5 µs** | **70.5** | 1.0× |
| Numba CUDA | 80.6 µs | 7.4 | 9.5× slower |
| *same-traffic floor* | *8.0 µs* | *75* | *0.94×* |
| *empty-kernel launch floor* | *6.7 µs* | — | — |

At 600 KB the "GB/s" figures are launch-overhead artifacts; the working set
stays in L2 across repetitions.

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test swiglu.cu test_harness.cu
```

`THREADS` can be overridden with `-D` (the default is 256).

Numba:

```bash
python3 swiglu.py
```

The script runs the correctness cases and the performance case.
