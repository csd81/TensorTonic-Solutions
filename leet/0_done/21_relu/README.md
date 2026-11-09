# ReLU — CUDA C++ and Numba Results

Challenge: `out[i] = max(0, in[i])` over a float vector.  `1 ≤ N ≤
100,000,000`; performance is measured at `N = 25,000,000`.

Files:

- `relu.cu` — CUDA C++ implementation of `solve(input, output, N)` (renamed
  from `starter.cu`).
- `relu_naive.cu` — the naive baseline: one thread per element, scalar
  loads/stores.
- `relu.py` — Numba CUDA port with the same signature.

## Where the time goes at N = 25,000,000

The kernel reads 100 MB and writes 100 MB — 200 MB of traffic with no reuse
and no arithmetic worth counting.  A pure `float4` **copy** of the same
200 MB measures **2.379 ms (84.1 GB/s)** on this GPU, and ReLU cannot be
faster than a copy.  The starter (one scalar element per thread) measures
**2.380 ms (84.0 GB/s)**: it is already at the floor.

So this challenge has essentially no headroom.  What the work below buys is a
version that is structured for the general case — vectorized when the
pointers allow it, a scalar kernel when they do not, and a correct tail for
`N % 4 != 0` — while matching the starter at the wall.

## Implementation — CUDA C++

One `float4` per thread, 512-thread blocks, one scalar tail element loop
handled in-kernel by the thread just past the last `float4`.  If `input` or
`output` is not 16-byte aligned, a scalar one-thread-per-element kernel runs
instead.  No grid-stride loop: with a huge grid (12 K blocks here) the
scheduler hides memory latency as well as it can be hidden, and the sweep
below shows grid-stride shapes are slightly slower.

Variant sweep (same traffic, one session — absolute values drift ~1% between
sessions on this GPU):

| Variant | Time | GB/s |
|---|---:|---:|
| scalar, 256 thr (starter shape) | 2.405 ms | 83.2 |
| scalar, 512 thr | 2.406 ms | 83.1 |
| float4 linear, 128 thr | 2.404 ms | 83.2 |
| float4 linear, 256 thr | 2.406 ms | 83.1 |
| float4 linear, 512 thr | 2.408 ms | 83.0 |
| float4 grid-stride, 1024 blocks × 256 thr | 2.436 ms | 82.0 |
| float4 grid-stride, 2 float4/thread | 2.437 ms | 82.1 |
| float4 grid-stride, 4 float4/thread | 2.440 ms | 82.0 |
| float4 grid-stride with `__ldcs`/`__stcs` | 2.426–2.441 ms | 81.9–82.4 |
| float4 packed, 2 adjacent float4/thread | 2.448 ms | 81.7 |
| float4 packed, 4 adjacent float4/thread | 2.608 ms | 76.7 |

Load width and block size do not matter at the wall; splitting the work into
fewer, longer-lived threads does.  Streaming cache hints were measured and
are not an improvement.

### Naive baseline

`relu_naive.cu` is the plainest correct form: one thread per element, plain
scalar load and store, and a branch (`v < 0 ? 0 : v`) instead of `fmaxf`.
It measures **2.376 ms (84.2 GB/s)** — the same as everything else here.
That is the honest result for this challenge: the naive mapping is *already*
the optimal mapping, because ReLU has no arithmetic to optimize and fully
coalesced scalar accesses saturate DRAM just as well as vectorized ones.  The
naive form is slower in compute-bound challenges, not in a pure streaming one.

## Implementation — Numba

Scalar one-element-per-thread kernel with a bounds check — Numba has no
`float4` type, and the measurement above shows the load width does not matter
here.  It reaches **2.445 ms (81.8 GB/s)**, about 2.9% behind the C++ kernel;
most of that is the per-launch overhead being visible in a 2.4 ms kernel.
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

Times are medians after a warm-up, including `cudaDeviceSynchronize()`.  The
three C++ variants were compiled against the same harness and run
interleaved on an idle GPU; run-to-run variation is about ±0.5%.

## Correctness — CUDA C++

ReLU is exact, so every case is checked for bit-exact equality against a CPU
reference:

| Case | Result |
|---|---|
| challenge example 1 (`[-2,-1,0,1,2]`) | exact |
| challenge example 2 (`[-3.5,0,4.2]`) | exact |
| single negative / single positive / single zero | exact |
| all negative / all positive / all zero | exact |
| sizes 2, 4, 5, 7, 1023, 1024, 1025, 4099 | exact |
| misaligned `input+1`, `output+1` (N = 1000) | exact |
| misaligned `input+2`, `output+3` (N = 1001) | exact |
| benchmark shape, N = 25,000,000 | exact |

## Correctness — Numba CUDA

All fifteen cases pass bit-exact against `np.maximum(a, 0)`, including the
challenge examples, odd sizes, and N = 25,000,000.

## Performance — N = 25,000,000

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| `relu_naive.cu` (one thread per element) | 2.376 ms | 84.2 | 1.00× |
| starter (scalar, one element per thread) | 2.380 ms | 84.0 | 1.00× |
| **`relu.cu`** (float4, 512-thread blocks) | **2.376 ms** | **84.2** | 1.0× |
| Numba CUDA | 2.445 ms | 81.8 | 1.03× slower |
| *float4 copy floor* | *2.379 ms* | *84.1* | *1.00×* |

Everything except the Numba port is the same number within noise: this
challenge is a DRAM-wall benchmark, and the starter was already at the wall.

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test relu.cu test_harness.cu
```

Numba:

```bash
python3 relu.py
```

The script runs the correctness cases and the performance case.  For best
performance, pass Numba device arrays; host arrays incur copies to and from
the device.
