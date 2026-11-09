# Value Clipping — CUDA C++ and Numba Results

Challenge: clip each element of a float32 vector to `[lo, hi]`
(`out[i] = min(max(in[i], lo), hi)`).  `1 ≤ N ≤ 100,000`; performance is
measured at `N = 100,000`.

Files:

- `value_clipping.cu` — CUDA C++ implementation of
  `solve(input, output, lo, hi, N)` (renamed from `starter.cu`).
- `value_clipping_naive.cu` — the naive baseline: one thread per element,
  scalar loads/stores and two comparisons.
- `value_clipping.py` — Numba CUDA port with the same signature.

## Where the time goes at N = 100,000

The problem moves only 800 KB (400 KB in, 400 KB out), so unlike the big
streaming challenges this one is **launch-bound**, not bandwidth-bound.  On
this GPU:

| Baseline | Time |
|---|---:|
| empty kernel launch + `cudaDeviceSynchronize()` | 6.7 µs |
| `float4` copy of the same 800 KB (traffic floor) | 8.6 µs |
| this ReLU-like clip kernel | 9.0 µs |

So ~75% of the kernel's time is the launch itself; the entire memory work
sits in the ~2 µs between the empty-kernel floor and the copy floor.

**The starter is not a usable baseline: its kernel body is empty.**  It
compiles and launches, but it neither reads nor writes anything, so every
correctness case fails.  Its measured 8.3 µs is the cost of launching 391
empty blocks and says nothing about the work.

## Implementation — CUDA C++

One `float4` per thread with 256-thread blocks (98 blocks at the benchmark
size), one kernel launch, and the scalar tail (`N % 4`) handled in-kernel by
the single thread just past the last `float4`.  If `input` or `output` is not
16-byte aligned, a scalar one-thread-per-element kernel runs instead.

Variant sweep (same 800 KB, medians of 200 launches; jitter is about ±0.5 µs):

| Variant | Time |
|---|---:|
| scalar, 256 thr (naive shape) | 10.5 µs |
| scalar, 512 thr | 10.5 µs |
| **float4, 128 thr** | **8.6 µs** |
| **float4, 256 thr** | **8.6 µs** |
| float4, 512 thr | 9.0–9.7 µs |
| float4, 1024 thr | 9.6–12.5 µs |
| float4 grid-stride, 25 blocks × 256 thr | 8.9 µs |
| float4 packed, 2 adjacent float4/thread | 11.4 µs |
| float4 packed, 4 adjacent float4/thread | 12.3 µs |

The scalar form pays ~2 µs (about 20%) for its 4× larger grid and 4× more
load/store instructions; the vectorized one is already at the copy floor.
Block sizes 64/128/256 tie when measured through the harness; 384/512 are
slightly slower.  Packing several float4 per thread hurts — 25,000 threads
is already only one wave, and serializing the accesses within a thread just
lengthens the critical path.

### Naive baseline

`value_clipping_naive.cu` is the plainest correct form: one thread per
element, plain scalar load/store, `if (v < lo) v = lo; if (v > hi) v = hi;`.
It measures **10.9 µs (73.4 GB/s)**, 1.21× slower than the vectorized kernel —
the largest relative gap of the small streaming challenges, because at 800 KB
there is no bandwidth to hide the per-thread instruction overhead.

## Implementation — Numba

Scalar one-element-per-thread kernel (Numba has no `float4` type).  It
measures **56.9 µs (14.1 GB/s)** — 6.3× the C++ kernel.  Most of that is
Python-side dispatch: an *empty* Numba kernel launch plus `cuda.synchronize()`
costs **21.5 µs**, and the remaining ~35 µs is the dispatcher marshalling five
arguments (two device arrays, two scalars, one int) plus a scalar
100,000-thread kernel.  At 800 KB there is nothing for the GPU to hide that
overhead behind.  Kernels use `cache=True`.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, 6 SMs) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Times are medians after a warm-up, including `cudaDeviceSynchronize()`.  The
C++ variants were compiled against the same harness and run sequentially on
an idle GPU.  At this problem size the run-to-run jitter is ~±0.5 µs (5%).

## Correctness — CUDA C++

Clipping is exact, so every case is checked for bit-exact equality against a
CPU reference:

| Case | Result |
|---|---|
| challenge example 1 (`[1.5,-2,3,4.5]`, `[0,3.5]`) | exact |
| challenge example 2 (`[-1,2,5]`, `[-0.5,2.5]`) | exact |
| single value clipped high / low / inside | exact |
| all inside / all below / all above | exact |
| `lo == hi`, full `[-1e6, 1e6]` range, range crossing zero | exact |
| sizes 2, 3, 5, 1023, 1025, 99,999, 100,000 | exact |
| misaligned `input+1`, `output+1` (N = 1000) | exact |
| misaligned `input+2`, `output+3` (N = 1001) | exact |
| benchmark shape | exact |

## Correctness — Numba CUDA

All eighteen cases pass bit-exact against `np.clip`, including the challenge
examples, `lo == hi`, odd sizes, and N = 100,000.

## Performance — N = 100,000

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| starter (stub: empty kernel, no valid output) | 8.3 µs | — | — |
| `value_clipping_naive.cu` (scalar, 1 thread/element) | 10.9 µs | 73.4 | 1.21× slower |
| **`value_clipping.cu`** (float4, 256-thread blocks) | **9.0 µs** | **88.9** | 1.0× |
| Numba CUDA | 56.9 µs | 14.1 | 6.3× slower |
| *float4 copy floor* | *8.6 µs* | *93.0* | *0.96×* |
| *empty-kernel launch floor* | *6.7 µs* | — | — |

The "GB/s" numbers at this size exceed what DRAM could sustain because the
800 KB working set stays in L2 across repetitions; the meaningful figures are
the absolute times and the gap to the launch floor.

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test value_clipping.cu test_harness.cu
```

`THREADS` can be overridden with `-D` (the default is 256).

Numba:

```bash
python3 value_clipping.py
```

The script runs the correctness cases and the performance case.  For best
performance, pass Numba device arrays; host arrays incur copies to and from
the device.
