# Multi-Head Self-Attention — CUDA C++ and Numba Results

Challenge: compute multi-head self-attention

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h)
head_i = softmax(Q_i K_i^T / sqrt(d_k)) V_i
```

with `Q, K, V` of size `N × d_model`, `d_k = d_model / h`, and each head
taking the contiguous `d_k`-wide column slice of the inputs.  Output is
`N × d_model`, row-major.  Constraints: `1 ≤ N ≤ 10,000`, `2 ≤ d_model ≤
1,024`, `1 ≤ h ≤ d_model`, `d_model % h == 0`.

The performance line specifies `N = 1,024` and `d_model = 1,024` but not
`h`, so the benchmark was measured for **h = 8 (d_k = 128), h = 16 (d_k = 64)
and h = 4 (d_k = 256)** and all three are reported.

Files:

- `multi_head_attention.cu` — CUDA C++ implementation of
  `solve(Q, K, V, output, N, d_model, h)` (renamed from `starter.cu`).
- `multi_head_attention.py` — Numba CUDA port with the same `solve`
  signature.

## Implementation — CUDA C++

**Fast path (`d_k ≤ 128`)** — flash-attention style, one pass over N:

| Parameter | Value |
|---|---:|
| Block | 256 threads (8 warps) |
| Rows of the head per warp | 2 |
| Keys per K/V tile (`BLOCK_N`) | 32 (one key per lane) |
| `d_k` elements per lane | up to 4 (`lane + 32·i`) |

- one warp owns two rows of the head's output; the 32 lanes are the 32 keys
  of the current K/V tile, so each lane computes one full score per row with
  a plain dot product — no cross-lane reduction for the scores;
- the running max / sum are warp-wide `__shfl_xor_sync` reductions;
- the head's K/V tile is staged in shared memory (padded by one float so
  `sK[lane][k]` and `sV[j][lane]` are bank-conflict free) and the next tile
  is prefetched into registers while the current one is used;
- `output = Concat(heads)` falls out of the column offsets: block `y` owns
  head `blockIdx.y`, whose slice starts at `blockIdx.y * d_k`.

**Fallback (`d_k > 128`)** — the starter's block-per-(head, row) kernel with
block-wide dot reductions is kept for heads wider than the shared-memory
tile supports; it is correct but no faster than the starter.

The starter was correct but slow (104 ms at h=8): it launches one block per
(head, query row), and every one of the N key iterations performs two
warp/block reductions and three `__syncthreads()`.

## Implementation — Numba

Identical structure: the flash kernel with scalar window registers, warp
shuffles and local prefetch buffers, plus the general fallback kernel using
static shared reduction arrays.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, sm_61) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-08 |

Kernel times are medians of 10 runs (C++) / 10 runs (Numba) after a warm-up,
with inputs on the device, including `cudaDeviceSynchronize()`.  Correctness
is checked against a float64 CPU reference (full reference for small cases,
sampled rows/columns for the benchmark size).

## Correctness — CUDA C++

| Case | N | d_model | h | Max abs error | Relative error |
|---|---:|---:|---:|---:|---:|
| challenge example 1 | 2 | 4 | 2 | 2.66e-07 | 6.64e-08 |
| challenge example 2 | 1 | 2 | 1 | 0.0 | 0.0 |
| one row | 1 | 8 | 8 | 0.0 | 0.0 |
| small | 5 | 8 | 4 | 1.04e-07 | 1.14e-07 |
| small | 7 | 16 | 2 | 6.59e-07 | 3.72e-07 |
| `d_k = 8` | 16 | 64 | 8 | 1.73e-07 | 1.30e-07 |
| `d_k = 32` | 33 | 128 | 4 | 4.06e-07 | 3.30e-07 |
| one element per head | 29 | 96 | 96 | 4.40e-07 | 2.47e-07 |
| `d_k = 256` (fallback) | 64 | 256 | 1 | 3.92e-07 | 2.65e-07 |
| wide model | 100 | 1024 | 8 | 1.11e-06 | 1.03e-06 |
| wide model | 64 | 1024 | 16 | 6.43e-07 | 4.34e-07 |
| `d_k = 256` (fallback) | 50 | 512 | 2 | 4.89e-07 | 4.47e-07 |

Benchmark outputs were re-verified after timing (relative error 9.1e-07 at
h=8, 6.7e-07 at h=16).

## Correctness — Numba CUDA

| Case | N | d_model | h | Max abs error | Relative error |
|---|---:|---:|---:|---:|---:|
| challenge example 1 | 2 | 4 | 2 | 2.66e-07 | 6.64e-08 |
| challenge example 2 | 1 | 2 | 1 | 0.0 | 0.0 |
| one row | 1 | 8 | 8 | 0.0 | 0.0 |
| small | 5 | 8 | 4 | 9.30e-08 | 9.20e-08 |
| small | 7 | 16 | 2 | 1.63e-07 | 1.55e-07 |
| `d_k = 8` | 16 | 64 | 8 | 3.55e-07 | 1.71e-07 |
| `d_k = 32` | 33 | 128 | 4 | 4.92e-07 | 4.69e-07 |
| one element per head | 29 | 96 | 96 | 4.15e-07 | 2.12e-07 |
| `d_k = 256` (fallback) | 64 | 256 | 1 | 3.88e-07 | 3.34e-07 |
| wide model | 100 | 1024 | 8 | 2.00e-06 | 1.82e-06 |
| wide model | 64 | 1024 | 16 | 7.93e-07 | 5.61e-07 |
| `d_k = 256` (fallback) | 50 | 512 | 2 | 3.61e-07 | 2.88e-07 |

## Performance — N = 1,024, d_model = 1,024

| h (d_k) | starter | CUDA C++ final | Numba CUDA |
|---|---:|---:|---:|
| **8 (128)** | 104.33 ms | **8.34–10.17 ms** (423–515 GFLOP/s) | 21.40 ms (201 GFLOP/s) |
| **16 (64)** | 112.10 ms | **12.67–13.56 ms** (317–339 GFLOP/s) | 31.58 ms (136 GFLOP/s) |
| 4 (256) | 112.93 ms | 112.4–114.4 ms (fallback) | 333.9 ms (fallback) |

The fast path is **10–12.5× faster than the starter** at h=8 and 8.3–8.8×
at h=16.  For h=4 the head is 256 wide, which does not fit the shared-memory
tile used by the fast path, so the kernel falls back to the starter's
algorithm and matches its speed (the Numba fallback is slower still).

`FLOPs = 4·N²·d_model` (the two `N²·d_k` products summed over all heads).

## Build / run

CUDA C++ (the file exports `solve`, so compile it with a harness that
provides `main`):

```bash
nvcc -O3 -arch=sm_61 -o test multi_head_attention.cu test_harness.cu
```

Numba:

```bash
python3 multi_head_attention.py
```

The Python script runs the correctness cases and the performance cases.  For
best performance, pass Numba device arrays; host arrays incur copies to and
from the device.
