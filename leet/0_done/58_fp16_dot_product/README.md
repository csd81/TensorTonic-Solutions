# FP16 Dot Product — CUDA C++ and Numba Results

Challenge: `result = Σ A[i]·B[i]` for FP16 vectors, FP32 accumulation, the
final value converted back to half.  `N ≤ 100,000,000`; benchmark
**N = 1e8** — 400 MB of reads and one half of output, so the kernel is purely
read-bound.  The pure-read rate on this GPU is ~87 GB/s, i.e. a ~4.6 ms floor.

Files:

- `fp16_dot_product.cu` — CUDA C++ implementation of `solve(A, B, result, N)`
  (renamed from `starter.cu`).
- `fp16_dot_product_naive.cu` — the naive baseline: the starter's kernel
  verbatim (scalar 2-byte half loads, shared tree reduction, per-call
  `cudaMalloc`, single-thread final reduction).
- `fp16_dot_product.py` — Numba CUDA port with the same signature.

## Results at N = 100,000,000

Same-process interleaved medians (20 rounds):

| Variant | Time | Effective GB/s |
|---|---:|---:|
| **naive = starter kernel** | 5.23 ms (5.13 standalone) | 76.5–78.0 |
| **shipped** | **4.64 ms** (4.62 standalone) | **86.2–86.6** |
| Numba CUDA port | 5.67 ms | 70.5 |

Speedup of the shipped kernel over the starter: **1.11–1.13×** — it runs at
the memory wall (~87 GB/s), so there is nothing further to win without
reading less data.

Correctness: both challenge examples (exact), `N = 1, 7, 8, 1000, 1e5, 2^20`,
and the benchmark shape all match a float64 reference.  Because the result is
stored as half, comparisons use a half-precision tolerance
(`1e-3·|ref| + 1e-2`); the GPU result is within one half-ulp of the rounded
reference (e.g. 3870 vs 3870.76 at the benchmark shape).

## Why the starter is slow, and what changed

The starter already streams the 400 MB correctly; its inefficiencies are:

- **Scalar 2-byte loads**: a warp's request covers 64 bytes, half a 128-byte
  sector, so it issues twice the transactions a vectorized kernel needs.
- A `cudaMalloc`/`cudaFree` pair per call and a single-thread final reduction
  over 2048 block sums.

The shipped kernel:

1. Loads **8 halves per thread per iteration** as one `float4` per input (a
   warp request is four 512-byte transactions), converts each `half2` with
   `__half22float2` and accumulates into **four independent FP32
   accumulators**.
2. Reduces the block with warp shuffles plus a shared partial array, then one
   `atomicAdd` per block into a cached single-float accumulator.
3. Converts to half in a 1-thread kernel, so a solve is three launches
   (4-byte memset, dot, convert) and allocates nothing.

Variants measured at N=1e8 (same process, interleaved):

| Variant | Time |
|---|---:|
| **float4 loads, 1024 blocks × 256 (shipped)** | **4.61 ms** |
| float4 loads, 8192 blocks | 4.62 ms |
| float4 loads, 4096 blocks | 4.65 ms |
| two float4 loads per iteration (16 halves) | 4.62 ms |
| half2 (4-byte) loads | 4.72 ms |
| float4 loads, 2048 blocks | 4.75 ms |

A scalar fallback handles inputs whose base pointers are not 16-byte aligned
(`cudaMalloc` always is, but the signature does not require it).

## Numba port

Numba cannot index `float16` arrays, so the port passes the buffers as
`int16` bit-pattern views and converts with `libdevice.half2float` /
`libdevice.float2half_rn`.  It also views the same buffers as `int32` so each
load fetches two halves (4 bytes per lane) which are split with shifts and
masks and fed to two FP32 FMAs; an odd-length tail is handled through the
`int16` view, and an odd-length buffer is sliced to even length before the
`int32` view (numpy/Numba cannot reinterpret an odd number of int16s as
int32).  Reduction is warp shuffles + one atomic per block.

Measured 5.67 ms (70.5 GB/s), 23% slower than the C++ kernel.
