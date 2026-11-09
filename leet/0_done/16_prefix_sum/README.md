# Prefix Sum — CUDA C++ and Numba Results

Challenge: inclusive prefix sum of a float array (`[a, b, c] → [a, a+b,
a+b+c]`).  `1 ≤ N ≤ 100,000,000`, values in `[-1000, 1000]`; performance is
measured at `N = 250,000`.

Files:

- `prefix_sum.cu` — CUDA C++ implementation of `solve(input, output, N)`
  (renamed from `starter.cu`).
- `prefix_sum_naive.cu` — the naive baseline: one thread per output summing
  everything before it.
- `prefix_sum.py` — Numba CUDA port with the same signature.

## Where the time goes at N = 250,000

The array is 1 MB, so this is not a bandwidth problem: the mandatory traffic
is 2 MB (~24 µs at this GPU's 87 GB/s) plus a ~7 µs launch/synchronise floor
measured with an empty kernel.  What decides the result is how many kernels
run and how much serial work sits between them.

The starter recurses over the block totals, with `cudaMalloc`/`cudaFree` at
every level and three kernels per level — for this size, block scan, recursive
scan of 489 block totals, add block offsets — and measures **66.2 µs**.

## Implementation — CUDA C++

One launch.  A decoupled-lookback scan (the CUB `DeviceScan` design): each
block scans its own 4096 elements (per-thread sequential runs of 16, a warp
scan of the run totals, a warp-0 scan of the per-warp totals), publishes its
block total, and resolves the exclusive prefix of its predecessors by walking
back over their published aggregates until it finds a block whose *inclusive*
prefix is already published.  Nothing else is needed: no second pass over the
data, no host recursion, no per-level allocation.  The lookback scratch is
cached across calls.

`ITEMS` (elements per thread) was swept: 4 → 57.0 µs, **16 → 43.0 µs**,
24 → 44.4 µs, 32 → 46.3 µs.  An ablation shows the load/store floor of this
kernel shape is ~34 µs and the lookback cascade costs ~9 µs; a cooperative
grid-sync alternative would trade it for two grid syncs, which measured
slower overall at this block count.

### Naive baseline

`prefix_sum_naive.cu` is the plainest correct form: one thread per output
element, each summing `input[0..i]` directly from global memory (O(N²) work,
no shared memory, no vector types, no warp intrinsics, no lookback).  It
passes the same harness — and is **~5,000× slower** (214.6 ms), which is what
the parallel scan exists to avoid.

## Implementation — Numba

The same single-launch lookback kernel: per-thread sequential runs, warp and
warp-0 scans via shuffles, and the decoupled lookback with an atomic status
word (a tiny kernel zeroes the status array first).  It reaches **158 µs** —
about 3.7× the C++ kernel, because a single Numba-compiled launch of this size
is dominated by its own launch and codegen overhead rather than bandwidth.

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
timings of the two C++ variants come from the same harness binary compiled
against each source.

## Correctness — CUDA C++

| Case | N | Result |
|---|---:|---|
| challenge example 1 (`[1,2,3,4]`) | 4 | exact |
| challenge example 2 (`[5,-2,3,1,-4]`) | 5 | exact |
| single element | 1 | exact |
| mixed signs | 2 | exact |
| all ones / all negative | 100 / 1,000 | exact |
| odd / block-sized / cross-block | 2,047 / 2,048 / 2,049 | ≤ 3.2e-03 abs |
| random | 100,000 | rel 3.9e-07 |
| benchmark size | 250,000 | rel 1.9e-07 |
| larger sizes (lookback cascades over many blocks) | 1,000,000 / 4,000,000 | rel ≤ 6.4e-07 |

The non-zero errors are float32 accumulation-order differences against the
float64 CPU reference; the naive sequential kernel is the most accurate of
the three (rel 5.5e-06 at the benchmark size is still well inside the 1e-4
tolerance used by the harness).

## Correctness — Numba CUDA

All eleven cases pass against `np.cumsum` in float64 (rel ≤ 4.7e-07).

## Performance — N = 250,000

| Implementation | Time | vs optimized |
|---|---:|---:|
| `prefix_sum_naive.cu` (O(N²) per-element scan) | 214.6 ms | 4,980× slower |
| starter (recursive multi-level scan) | 66.2 µs | 1.54× slower |
| **`prefix_sum.cu`** (single-launch decoupled lookback) | **43.0 µs** | 1.0× |
| Numba CUDA | 158.2 µs | 3.7× slower |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test prefix_sum.cu test_harness.cu
```

`ITEMS` can be overridden with `-D` (the default is 16).

Numba:

```bash
python3 prefix_sum.py
```

The script runs the correctness cases and the performance case.  For best
performance, pass Numba device arrays; host arrays incur copies to and from
the device.
