# Dot Product — CUDA C++ and Numba Results

Challenge: dot product of two float vectors, `result = Σ A[i]·B[i]`, with the
result written to `result[0]`.  `1 ≤ N ≤ 100,000,000`; performance is
measured at `N = 4,194,304`.

Files:

- `dot_product.cu` — CUDA C++ implementation of `solve(A, B, result, N)`
  (renamed from `starter.cu`).
- `dot_product_naive.cu` — the naive baseline: one thread per element with an
  atomicAdd into the result.
- `dot_product.py` — Numba CUDA port with the same signature.

## Where the time goes

The kernel must read both vectors — 32 MB at the benchmark size — and writes
four bytes, so it is purely a bandwidth question: 32 MB at this GPU's
practical 87 GB/s is ~0.39 ms.  The starter (scalar grid-stride loads, a
warp/block reduction and one atomic per block) measures **0.397 ms — 84.5
GB/s, 97% of the roofline**.  It is already the right kernel.

The final version therefore only changes the mechanics: `float4` loads (512 B
per warp instruction instead of 128 B), the same warp/block reduction, one
atomic per block, and a scalar grid-stride tail for `N` not divisible by 4.
The measured time is identical within run-to-run noise (0.3969 vs 0.3972 ms)
— which is what a bandwidth-bound kernel at 97% of peak looks like.  An
alignment check falls back to scalar loads if the pointers are not 16-byte
aligned.

### Naive baseline

`dot_product_naive.cu` is the plainest correct form: one thread per element
with `atomicAdd(result, A[i] * B[i])` — no shared memory, no warp intrinsics,
no vector types, no grid-stride loop.  It is **21.7× slower (8.61 ms, 3.9
GB/s)** because four million atomics into one address serialise, and it is
also *less accurate*: summing every product into a single float32 gives a
relative error of 3.7e-05 at the benchmark size, against 3.6e-06 for the
block-reduced version (the harness tolerance is 1e-4).

## Implementation — Numba

The same shape: grid-stride loop, per-thread accumulation, warp/block
reduction through `shfl_down_sync` and shared memory, one atomic per block.
It reaches **0.538 ms (62 GB/s)**, 1.36× the C++ kernel — Numba's scalar
loads move 128 B per warp instruction instead of 512 B, which is the whole
difference on a bandwidth-bound kernel.  Kernels use `cache=True` so the JIT
cost is paid once per source change.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, 6 SMs) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Times are medians of 30–50 runs after a warm-up, including
`cudaDeviceSynchronize()`.  The two C++ variants were timed from the same
harness binary compiled against each source.

## Correctness — CUDA C++

Checked against a float64 CPU dot product (tolerance `1e-4 · max(1, |ref|)`):

| Case | N | optimized | naive |
|---|---:|---|---|
| challenge example 1 (`[1,2,3,4]·[5,6,7,8] = 70`) | 4 | exact | exact |
| challenge example 2 (`= 15.5`) | 3 | exact | exact |
| single element | 1 | exact | exact |
| cancelling terms | 2 | exact | exact |
| random | 1,000 | rel 9e-07 | rel 1e-06 |
| random | 100,000 | rel 6e-07 | rel 4e-06 |
| benchmark size | 4,194,304 | rel 3.6e-06 | rel 3.7e-05 |

## Correctness — Numba CUDA

All seven cases pass against a float64 `np.dot`, worst relative error
9.9e-07.

## Performance — N = 4,194,304

| Implementation | Time | Bandwidth | vs optimized |
|---|---:|---:|---:|
| `dot_product_naive.cu` (one atomic per element) | 8.61 ms | 3.9 GB/s | 21.7× slower |
| starter (scalar loads, block reduction) | 0.397 ms | 84.5 GB/s | 1.00× |
| **`dot_product.cu`** (float4 loads) | **0.3969 ms** | **84.6 GB/s** | 1.00× |
| Numba CUDA | 0.538 ms | 62.3 GB/s | 1.36× slower |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test dot_product.cu test_harness.cu
```

`THREADS` and `MAX_BLOCKS` can be overridden with `-D` (the defaults are 256
and 1024).

Numba:

```bash
python3 dot_product.py
```

The script runs the correctness cases and the performance case.  For best
performance, pass Numba device arrays; host arrays incur copies to and from
the device.
