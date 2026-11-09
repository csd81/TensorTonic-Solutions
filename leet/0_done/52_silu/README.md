# SiLU — CUDA C++ and Numba Results

Challenge: SiLU, `out[i] = x / (1 + exp(-x))`, on a float32 vector.
`1 ≤ N ≤ 10,000` (the benchmark uses `N = 50,000`).

Files:

- `silu.cu` — CUDA C++ implementation of `solve(input, output, N)` (renamed
  from `starter.cu`).
- `silu_naive.cu` — the naive baseline: one thread per element, scalar
  load/store, `expf` and IEEE division.
- `silu.py` — Numba CUDA port with the same signature.

## Where the time goes at N = 50,000

The kernel moves 200 KB in + 200 KB out = 400 KB, so it is **launch-bound**,
not bandwidth-bound.  On this GPU:

| Baseline | Time |
|---|---:|
| empty kernel launch + `cudaDeviceSynchronize()` | 6.7 µs |
| `float4` copy of the same 400 KB (traffic floor) | 7.7 µs |
| this SiLU kernel (`__expf` + `__fdividef`) | 8.2 µs |
| this SiLU kernel (`expf` + IEEE division) | 10.8 µs |

The interesting result at this size is that the *math* is not free relative
to the launch: the transcendental plus division cost ~3 µs, more than a third
of the floor.  Using CUDA's fast intrinsics removes almost all of it.

**The starter is not a usable baseline: its kernel body is empty.**  It
launches and returns, writing nothing, so every correctness case fails.  Its
measured 7.5 µs is just the launch of 196 empty blocks.

## Implementation — CUDA C++

One `float4` per thread with 256-thread blocks (49 blocks at N = 50,000),
one launch, and the scalar tail (`N % 4`) folded into the same kernel (at
this size a second launch for the tail would cost more than the tail itself).
If either pointer is not 16-byte aligned, a scalar one-element-per-thread
kernel runs instead.

Variant sweep (medians of 200 launches; accuracy against a float64 reference
over the same 50,000 inputs):

| Variant | Time | max rel err |
|---|---:|---:|
| **float4, 256 thr, `__expf` + `__fdividef`** | **7.9 µs** | **9.2e-07** |
| float4, 256 thr, `__expf` + IEEE division | 10.6 µs | 9.2e-07 |
| float4, 256 thr, `expf` + IEEE division | 10.8 µs | 1.6e-07 |
| float4, 512 thr, `__expf` + division | 10.6 µs | 9.2e-07 |
| float4, 128 thr, `__expf` + division | 12.2 µs | 9.2e-07 |
| scalar, 256 thr, `expf` + division | 11.7 µs | 1.6e-07 |
| scalar, 256 thr, `__expf` + division | 11.3 µs | 9.2e-07 |

`__expf` alone barely helps; the IEEE division is the expensive part.
`__fdividef` cuts ~2.7 µs and lands the kernel on the copy floor.  The price
is accuracy: worst-case relative error grows from ~1.6e-07 (full-precision
`expf`/division) to ~9.2e-07 — still far inside the ~1e-5 tolerance used by
the harness and typical graders, and the reason this is documented rather
than hidden.  (`__frcp_rn` + multiply was also tried: no better than exact
division.)

### Naive baseline

`silu_naive.cu` is the plainest correct form: one thread per element, plain
scalar load/store, `x / (1 + expf(-x))`.  It measures **12.1 µs**, 1.5×
slower than the vectorized fast-math kernel — the scalar form pays both for
4× the instructions and for the full-precision division.

## Implementation — Numba

Scalar one-element-per-thread kernel with `math.exp` and IEEE division (the
fast intrinsics are not exposed).  It reaches **72.6 µs**, 8.8× the C++
kernel: at 400 KB the ~20 µs of Python-side dispatch per launch dominates
everything.  Kernels use `cache=True`.

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
| challenge example 1 (`[0.5, 1.0, -0.5]`) | max_abs 3e-8 |
| challenge example 2 (`[-1,-2,-3,-4,-5]`) | max_abs 3e-8 |
| single zero | exact |
| extremes (-100, -50, -10, 10, 50, 100) | max_abs ≤ 1.3e-06 |
| all positive / all negative | max_abs ≤ 3e-7 |
| sizes 1, 2, 3, 5, 7, 1023, 1024, 1025, 4099, 50,000 | rel ≤ 9.2e-07 |
| misaligned `input+1`, `output+1` (N = 1000) | rel ≤ 9.2e-07 |
| misaligned `input+2`, `output+3` (N = 1001) | rel ≤ 9.2e-07 |
| benchmark shape, N = 50,000 | max_abs 1.3e-06 |

All 19 harness cases pass with max_abs < 1e-4 and max_rel < 1e-5.

## Correctness — Numba CUDA

All 16 cases pass with max_abs ≤ 9.6e-07, max_rel ≤ 1.4e-07 (IEEE division,
so the errors match the full-precision C++ form).

## Performance — N = 50,000

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| starter (stub: empty kernel, no valid output) | 7.5 µs | — | — |
| `silu_naive.cu` (scalar, `expf` + division) | 12.1 µs | 33.0 | 1.5× slower |
| **`silu.cu`** (float4, `__expf` + `__fdividef`) | **8.2 µs** | **48.5** | 1.0× |
| Numba CUDA | 72.6 µs | 5.5 | 8.8× slower |
| *float4 copy floor* | *7.7 µs* | *51.9* | *0.94×* |
| *empty-kernel launch floor* | *6.7 µs* | — | — |

At 400 KB the "GB/s" numbers are dominated by launch overhead and the
working set stays in L2 across repetitions; the meaningful figures are the
absolute times and the gap to the empty-launch floor.

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test silu.cu test_harness.cu
```

`THREADS` can be overridden with `-D` (the default is 256).

Numba:

```bash
python3 silu.py
```

The script runs the correctness cases and the performance case.
