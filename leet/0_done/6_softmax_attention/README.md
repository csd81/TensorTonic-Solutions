# Softmax Attention — CUDA C++ and Numba Results

Challenge: compute

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
```

with the softmax applied row-wise, for `Q` of size `M×d`, `K` and `V` of size
`N×d`, and `1 ≤ M, N ≤ 100,000`, `1 ≤ d ≤ 128`.  Performance is measured at
`M = 512`, `N = 256`.

Files:

- `softmax_attention.cu` — CUDA C++ implementation of
  `solve(Q, K, V, output, M, N, d)` (renamed from `starter.cu`).
- `softmax_attention.py` — Numba CUDA port with the same `solve` signature.

## Implementation

Both versions implement **FlashAttention-style fused attention**: one kernel
that walks over `K`/`V` in tiles and maintains the online-softmax state
`(m_i, l_i, out_i)`, so the `M×N` score matrix is never materialised.  Peak
extra memory is `O(M·d + N·d)`.

Kernel layout (CUDA C++ and Numba are identical):

| Parameter | Value |
|---|---:|
| Threads per block | 256 (8 warps) |
| Rows of `Q` per warp | 2 |
| Rows of `Q` per block | 16 |
| Keys per tile (`BLOCK_N`) | 32 (one key per lane) |
| `d` per thread in `P·V` | up to 4 (`lane + 32·i`) |

Per tile, each lane:

1. computes the full dot product of its key against the warp's two `Q` rows
   (`sK[lane][k]` is bank-conflict free, `sQ` is a broadcast),
2. participates in a warp-wide `__shfl_xor_sync` max/sum reduction for the
   online-softmax update,
3. accumulates `P·V` for both rows, reusing each `sV` load for both rows.

Shared memory (43,328 B at `d = 128`): `sQ` 16×(128+1), `sK` and `sV`
32×(128+1) each, plus `sP` 16×32.  The `+1` padding makes both
`sK[lane][k]` and `sV[j][lane]` conflict-free.

### Software pipelining

Tile `t+1` is loaded from global memory into registers while tile `t` is
being computed (32 floats per thread), so the DRAM latency is overlapped with
the score and `P·V` work.  This cut the kernel from 0.209 ms to 0.166 ms at
the benchmark size.

### Changes from the starter

The starter was verified correct on all test cases, but very slow
(2.899 ms at the benchmark size), because:

- it ran **one warp per block** (32 threads, 16 blocks), leaving the GPU
  almost idle;
- `float q_reg[MAX_D]` and `float out_reg[MAX_D]` are indexed dynamically, so
  they spill to local memory (≈2 KB of stack per thread);
- every block re-read all of `K` and `V`.

The final kernel keeps the same online-softmax algorithm but uses 8 warps per
block, keeps `Q` in shared memory, stages `K`/`V` tiles once per block,
uses warp-shuffle reductions, and pipelines the tile loads.  Net result:
**2.899 ms → 0.165 ms (≈17.6×)**.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, sm_61) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-08 |

Kernel times are medians after a warm-up, with `Q`, `K`, `V` and the output
already on the device.  The timed region includes `cudaDeviceSynchronize()`.
C++ uses 30 runs, Numba 20.  Correctness is checked against a float64 CPU
reference (`Q @ K.T / sqrt(d)`, row-wise softmax, `@ V`).

## Correctness — CUDA C++

| Case | M | N | d | Max abs error | Relative error |
|---|---:|---:|---:|---:|---:|
| challenge example 1 | 2 | 3 | 4 | 4.77e-07 | 5.96e-08 |
| challenge example 2 | 1 | 2 | 2 | 2.60e-07 | 4.87e-08 |
| single element | 1 | 1 | 1 | 0.0 | 0.0 |
| odd shapes | 33 | 33 | 3 | 1.95e-07 | 2.23e-07 |
| small | 7 | 5 | 4 | 7.85e-08 | 4.51e-08 |
| `d = 128` | 100 | 37 | 128 | 6.40e-07 | 5.56e-07 |
| non-multiples | 65 | 100 | 64 | 3.96e-07 | 5.04e-07 |
| larger N | 300 | 1000 | 128 | 4.74e-07 | 1.97e-06 |
| long N | 64 | 4096 | 64 | 2.23e-07 | 2.22e-06 |
| many rows | 2000 | 64 | 32 | 4.60e-07 | 4.11e-07 |
| partial block | 17 | 33 | 7 | 1.60e-07 | 2.03e-07 |

A sweep of `d ∈ {1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 33, 63, 65, 96, 100, 127,
128}` at `M=35, N=47` passed with relative error ≤ 6.7e-07.  The two challenge
examples reproduce the published outputs (`4.29 5.29 6.29 7.29` / `5 6 7 8`
and `4.34 5.34`).

## Correctness — Numba CUDA

| Case | M | N | d | Max abs error | Relative error |
|---|---:|---:|---:|---:|---:|
| challenge example 1 | 2 | 3 | 4 | 4.77e-07 | 5.96e-08 |
| challenge example 2 | 1 | 2 | 2 | 2.60e-07 | 4.87e-08 |
| single element | 1 | 1 | 1 | 0.0 | 0.0 |
| small | 7 | 5 | 4 | 8.70e-08 | 1.35e-07 |
| odd shapes | 33 | 33 | 3 | 1.84e-07 | 2.91e-07 |
| partial block | 17 | 33 | 7 | 1.27e-07 | 1.96e-07 |
| non-multiples | 65 | 100 | 64 | 5.10e-07 | 6.57e-07 |
| `d = 128` | 100 | 37 | 128 | 5.89e-07 | 4.84e-07 |
| larger N | 300 | 1000 | 128 | 3.89e-07 | 1.69e-06 |
| long N | 64 | 4096 | 64 | 2.32e-07 | 2.25e-06 |
| many rows | 2000 | 64 | 32 | 5.17e-07 | 3.88e-07 |

The host-array path (2-D NumPy inputs) was also tested and matched with max
absolute error 1.59e-07.

## Performance

`FLOPs = 4·M·N·d` (two `M×N×d` products).  Device-resident inputs, median of
20–30 runs.

| M | N | d | CUDA C++ | Numba CUDA |
|---:|---:|---:|---:|---:|
| **512** | **256** | **128** | **0.165 ms** (407 GFLOP/s) | 0.568 ms (118 GFLOP/s) |
| 512 | 256 | 64 | 0.125 ms (268 GFLOP/s) | 0.455 ms (74 GFLOP/s) |
| 2048 | 1024 | 128 | 2.280 ms (471 GFLOP/s) | 5.596 ms (192 GFLOP/s) |

The final CUDA kernel is 17.6× faster than the starter (2.899 ms) at the
challenge's measurement size.  Numba is about 3.4× slower than the hand
written CUDA, mostly because its local arrays (`kbuf`/`vbuf`) live in local
memory rather than registers, so the prefetch pipeline is less effective.

### Variants benchmarked (M=512, N=256, d=128)

| Variant | Time |
|---|---:|
| starter (1 warp/block, 128-float local arrays) | 2.899 ms |
| warp-per-row, 2 rows/warp, K/V in shared | 0.209 ms |
| same, V read from global (3 blocks/SM) | 0.203 ms |
| 4 rows/warp, `float4`-packed Q, 4 warps/block | 0.275 ms |
| 4 rows/warp, `float4` Q, global V, 8 warps/block | 0.249 ms |
| **final: 2 rows/warp + register-prefetch pipeline** | **0.165 ms** |

Notes from the tuning process:

- More rows per warp reduces per-row shared-memory traffic, but the larger
  register footprint and lower warp count cost more than it saves on this
  GPU.
- `float2`/`float4` packing of the `Q` broadcasts is neutral: a microbenchmark
  on this GPU shows `LDS.128` costs ~3.4× an `LDS.32` per instruction, so
  vectorising broadcast loads buys nothing.
- The register-prefetch pipeline is the single biggest win after the layout
  change.

## Build / run

CUDA C++ (the file exports `solve`, so compile it with a harness that
provides `main`):

```bash
nvcc -O3 -arch=sm_61 -o test softmax_attention.cu test_harness.cu
```

Numba:

```bash
python3 softmax_attention.py
```

The Python script runs the correctness cases and the performance cases.  For
best performance, pass Numba device arrays; host arrays incur per-kernel
copies.  `d > 128` is rejected by both implementations (the challenge
guarantees `d ≤ 128`).

## Naive baseline

`softmax_attention_naive.cu` is the same computation written as plainly as possible:
one thread per output element, two passes over the N keys recomputing the scaled dot product from global memory each time — no shared-memory tiling, no vector types, no warp intrinsics and no
fast-math.  It passes the same correctness cases as the optimized kernel
(checked against a float64 CPU reference in the shared harness).

| Implementation | Time | vs naive |
|---|---:|---:|
| `softmax_attention_naive.cu` | 60.822 ms | 1.00x |
| `softmax_attention.cu` (optimized) | 0.1503 ms | 405x |

The optimized kernel streams the keys once in a flash-attention style pass with the row state in registers.
