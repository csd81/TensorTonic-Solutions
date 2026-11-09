# Interleave — CUDA C++ and Numba Results

Challenge: interleave two float32 arrays `A` and `B` (each length `N`) into
an output of length `2N`: `out[2i] = A[i]`, `out[2i+1] = B[i]`.
`1 ≤ N ≤ 50,000,000`; performance is measured at `N = 25,000,000`.

Files:

- `interleave.cu` — CUDA C++ implementation of `solve(A, B, output, N)`
  (renamed from `starter.cu`).
- `interleave_naive.cu` — the naive baseline: one thread per input index,
  two scalar stores.
- `interleave.py` — Numba CUDA port with the same signature.

## Where the time goes at N = 25,000,000

The kernel reads 200 MB and writes 200 MB — **400 MB** of traffic.  A plain
`float4` copy of the same 400 MB measures **4.82–4.86 ms (82–83 GB/s)**, and
the starter's scalar interleave measures **4.85 ms**: it is already at the
wall.

The starter has an apparent flaw worth testing: each thread writes
`out[2i]` and `out[2i+1]` with two 4-byte stores at an 8-byte stride, so a
warp's single store instruction touches every other float over a 256-byte
range.  At DRAM speed that costs nothing — the measured time is the copy
floor.  Vectorized interleaves were implemented anyway to confirm:

| Variant | Time | GB/s |
|---|---:|---:|
| **scalar, 512 thr (shipped)** | **4.81–4.85 ms** | **83** |
| scalar, 256 thr (starter/naive shape) | 4.80–4.88 ms | 82–83 |
| float2 loads → float4 store, 128/256/512 thr | 4.896–4.903 ms | 81.6 |
| float4 loads → 2× float4 stores, 128/256/512 thr | 4.90–4.92 ms | 81.4–81.7 |

The vectorized forms are 0.4–1% *slower*: they cut instruction count (not the
bottleneck) and add register pressure, while DRAM traffic is unchanged.  The
shipped kernel is the plain scalar mapping with 512-thread blocks.

## Implementation — CUDA C++

One thread per input index, two `__ldg` loads and two scalar stores,
512-thread blocks, one launch.  No vector types, no alignment checks — scalar
accesses are valid for any pointer and any `N`.

### Naive baseline

`interleave_naive.cu` is the plainest correct form: one thread per index,
two scalar loads and stores.  It measures **4.80 ms (83.4 GB/s)** — the same
as the shipped kernel within noise (it was the fastest single run in one
session, which is what ±1% noise looks like here).  As with the other pure
streaming challenges, the naive mapping is already the optimal mapping.

## Implementation — Numba

Same scalar mapping (Numba has no `float4` type).  It reaches **5.05 ms
(79.2 GB/s)**, 4.8% behind the C++ kernel — at 400 MB, per-launch overhead is
negligible and the gap is codegen.  Kernels use `cache=True`.

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
an idle GPU.  Run-to-run variation is about ±1%.

## Correctness — CUDA C++

The interleave is exact, so every case is checked for bit-exact equality:

| Case | Result |
|---|---|
| challenge example 1 (`[1,2,3]` / `[4,5,6]`) | exact |
| challenge example 2 (`[10,20]` / `[30,40]`) | exact |
| N = 1 | exact |
| sizes 2, 3, 5, 7, 1023, 1024, 1025, 4099, 100,000 | exact |
| misaligned `A+1`, `B+1`, `output+1` (N = 1000) | exact |
| misaligned `A+2`, `B+3`, `output+2` (N = 1001) | exact |
| benchmark shape, N = 25,000,000 | exact |

## Correctness — Numba CUDA

All twelve cases pass bit-exact, including both examples and N = 25,000,000.

## Performance — N = 25,000,000

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| `interleave_naive.cu` (scalar, 256 thr) | 4.80 ms | 83.4 | 1.00× |
| starter (scalar, 256 thr) | 4.85 ms | 82.6 | 1.01× |
| **`interleave.cu`** (scalar, 512 thr) | **4.82 ms** | **83.1** | 1.0× |
| Numba CUDA | 5.05 ms | 79.2 | 1.05× slower |
| *float4 copy floor* | *4.85 ms* | *82.5* | *1.01×* |

Everything on the GPU side is the same number within noise: this challenge
is the memory wall itself.

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test interleave.cu test_harness.cu
```

`THREADS` can be overridden with `-D` (the default is 512).

Numba:

```bash
python3 interleave.py
```

The script runs the correctness cases and the performance case.
