# Batched Matrix Multiplication — CUDA C++ and Numba Results

Challenge: `C_b = A_b · B_b` for `BATCH` batches; `A [BATCH, M, K]`,
`B [BATCH, K, N]`, `C [BATCH, M, N]`, all row-major FP32.
`BATCH ≤ 128`, `M, N, K ≤ 1024`.  The benchmark shape is stated as
**M = N = K = 256**; the challenge does not state the benchmark batch size,
so all of B = 1, 8, 32, 128 are reported.

Files:

- `batched_matrix_multiplication.cu` — CUDA C++ implementation of
  `solve(A, B, C, BATCH, M, N, K)` (renamed from `starter.cu`).
- `batched_matrix_multiplication_naive.cu` — the naive baseline: the
  starter's 16×16-tile, one-output-per-thread kernel verbatim.
- `batched_matrix_multiplication.py` — Numba CUDA port.

## Results at M = N = K = 256

Same-process interleaved medians:

| BATCH | naive = starter | shipped | speedup | Numba port |
|---:|---:|---:|---:|---:|
| 1 | 0.119 ms | **0.065 ms** | 1.84× | 0.203 ms |
| 8 | 0.889 ms | **0.421 ms** | 2.11× | 0.993 ms |
| 32 | 3.510 ms | **1.538 ms** | 2.28× | 3.754 ms |
| 128 | 14.476 ms | **6.015 ms** | 2.41× | 14.355 ms |

Throughput: 0.52–0.71 TFLOP/s for the shipped kernel against 0.28–0.31 for the
starter.

Correctness: the challenge example and random cases `1x1x1x1`, `3x5x7x11`,
`7x17x33x19`, `5x64x128x64`, `2x129x130x131` (checked element-for-element in
double) plus the benchmark shape at every batch size (sampled, since
B=128 holds 8.4 M outputs).  Tolerance `1e-3 + 1e-4·|ref|`.

## Why the starter is slow, and what changed

The starter uses a 16×16 shared tile with one output per thread — two shared
loads per multiply-add — and one `grid.z` slice per batch.  The shipped
kernel is the register-blocked GEMM family: **64×64 block tile, 16×16
threads, 4×4 per thread** (0.5 shared loads per multiply-add), one grid.z
slice per batch.

The tile size is the interesting choice here: at M=N=K=256 the 128×128 tile
with 8×8 per thread — which wins on larger matrices — only produces four
blocks per batch and loses badly at every batch size:

| Shape | B=1 | B=8 | B=32 | B=128 |
|---|---:|---:|---:|---:|
| **64×64, 4×4, 16×16 thr (shipped)** | **0.52** | **0.63** | **0.70** | **0.71** TFLOP/s |
| 64×128, 4×8, 16×16 thr | 0.38 | 0.58 | 0.65 | 0.63 TFLOP/s |
| 128×64, 8×4, 16×16 thr | 0.41 | 0.47 | 0.51 | 0.49 TFLOP/s |
| 128×128, 8×8, 16×16 thr | 0.32 | 0.35 | 0.38 | 0.38 TFLOP/s |

## Numba port

Same structure at a Numba-friendly size: 64×64 tile, 16×16 threads, 4×4 per
thread with 16 explicit scalar accumulators and int32 block/index arithmetic.

A bug worth recording: the port's store guards for the second/third/fourth
row of each thread tile initially compared against `N` instead of `M` (the
square-matrix port this was adapted from used one symbol for both).  With
M < N the out-of-range rows then carried *zero* accumulators into the **next
batch's** slice, so batch b's rows 0–1 were raced against zeros from batch
b−1's block — the failures only appeared for non-square M≠N cases and moved
with the batch count.  The guards now compare against `M`.

Measured 0.203 / 0.993 / 3.754 / 14.355 ms for B = 1/8/32/128 (0.17–0.30
TFLOP/s) — only on par with the C++ starter at large batches, because the
Numba GEMM reaches ~0.3 TFLOP/s.
