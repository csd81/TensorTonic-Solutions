# FP16 Batched Matrix Multiplication — CUDA C++ and Numba Results

Challenge: `C_b = A_b · B_b` for `BATCH` batches in FP16, FP32 accumulation,
half output.  `A [BATCH, M, K]`, `B [BATCH, K, N]`, `C [BATCH, M, N]`, all
row-major.  `BATCH ≤ 128`, `M, N, K ≤ 1024`; the benchmark shape is
**M = N = K = 256** and the batch size is not stated, so all of B = 1, 8, 32,
128 are reported.

Files:

- `fp16_batched_matmul.cu` — CUDA C++ implementation of
  `solve(A, B, C, BATCH, M, N, K)` (renamed from `starter.cu`).
- `fp16_batched_matmul_naive.cu` — the naive baseline: the starter's kernel
  verbatim (32×32 half shared tiles, one output per thread, both operands
  converted to float in every multiply-add).
- `fp16_batched_matmul.py` — Numba CUDA port.

## Results at M = N = K = 256

Same-process interleaved medians:

| BATCH | naive = starter | shipped | speedup | Numba port |
|---:|---:|---:|---:|---:|
| 1 | 0.142 ms | **0.066 ms** | 2.15× | 0.192 ms |
| 8 | 1.059 ms | **0.416 ms** | 2.55× | 0.912 ms |
| 32 | 4.178 ms | **1.543 ms** | 2.71× | 3.410 ms |
| 128 | 17.191 ms | **6.074 ms** | 2.83× | 13.559 ms |

Throughput: 0.51–0.71 TFLOP/s shipped against 0.24–0.26 for the starter.

Correctness: the challenge example and random cases `1x1x1x1`, `3x5x7x11`,
`7x17x33x19`, `5x64x128x64`, `2x129x130x131` (element-for-element in double)
plus the benchmark shape at every batch size (sampled).  The output is half,
so the tolerance is `1e-3·|ref| + 1e-2` (max observed absolute error 7.8e-3
on outputs of magnitude ~50–100).

## Why the starter is slow, and what changed

The starter keeps its shared tiles in *half* and converts both operands to
float **inside the inner loop**, so every multiply-add pays two conversions.
It also uses a 16×16/32×32 tile with one output per thread (two shared loads
per multiply-add).

The shipped kernel follows the FP16 GEMM from challenge 22:

1. **The shared tiles hold float**, converted once when each element is
   loaded.  Each tile element is consumed BK−1 more times in the
   micro-kernel than in the loader, so this removes ~30× the conversions.
2. 64×64 block tile, 16×16 threads, 4×4 per thread (0.5 shared loads per
   multiply-add), one `grid.z` slice per batch.
3. FP32 accumulators, half epilogue.

Shape sweep at M=N=K=256 (TFLOP/s):

| Shape | B=1 | B=8 | B=32 | B=128 |
|---|---:|---:|---:|---:|
| **64×64, 4×4, 16×16 thr (shipped)** | **0.51** | **0.64** | 0.70 | 0.71 |
| 128×128, 8×8, 16×16 thr | 0.31 | 0.63 | **0.71** | **0.73** |
| 64×128, 4×8, 16×16 thr | 0.38 | 0.59 | 0.66 | 0.70 |
| 128×64, 8×4, 16×16 thr | 0.41 | 0.48 | 0.54 | 0.55 |

The 128×128 tile only overtakes at B ≥ 32 (and by ~2%); at B=1 it is 60%
slower because M=N=K=256 leaves it four blocks per batch.  The 64×64 shape is
shipped as the robust choice.

## Numba port

Numba cannot index `float16` arrays, so the buffers are passed as `int16`
bit-pattern views and converted with `libdevice.half2float` (tile load) and
`libdevice.float2half_rn` (epilogue).  Same 64×64/4×4 tile, 16 scalar
accumulators, `grid.z` per batch and int32 index arithmetic (the same port
family as challenge 30, where the row guards must compare against `M` — not
`N` — or out-of-range rows spill into the next batch's slice).

Measured 0.192 / 0.912 / 3.410 / 13.559 ms (0.18–0.32 TFLOP/s), only ~1.25×
faster than the C++ starter at large batches.
