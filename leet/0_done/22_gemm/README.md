# GEMM (FP16 in, FP32 accumulate) — CUDA C++ and Numba Results

Challenge: `C = alpha·(A×B) + beta·C` with `A` (M×K), `B` (K×N) and `C`
(M×N) row-major `half`, `alpha`/`beta` float; accumulation must be FP32 and
the result stored as `half`.  `16 ≤ M, N, K ≤ 4096`; performance is measured
at `M = N = K = 1024`.

Files:

- `gemm.cu` — CUDA C++ implementation of
  `solve(A, B, C, M, N, K, alpha, beta)` (renamed from `starter.cu`).
- `gemm_naive.cu` — the naive baseline: 32×32 shared-memory tile, one output
  element per thread (the starter's shape).
- `gemm.py` — Numba CUDA port with the same signature.

## Where the time goes

At 1024³ the problem is 2.15 GFLOP on 6 MB of matrices — **compute-bound**.
This GPU has no tensor cores (compute capability 6.1 predates WMMA), so the
ceiling is the FP32 pipeline: 6 SMs × 128 lanes × 2 FLOP × ~1.48 GHz ≈
**2.27 TFLOP/s**.

The starter's 32×32-tile, one-output-per-thread kernel reads both operands
from shared memory for every FMA (2 shared loads per FMA) and is
shared-memory-bandwidth-bound: **7.72 ms, 0.278 TFLOP/s** — about 12% of
peak.  The shipped kernel is **2.07 ms, 1.04 TFLOP/s (~46% of peak)**, a
**4.0×** improvement.

## Implementation — CUDA C++

- **128×128 block tile, 16×16 threads, 8×8 per thread** (64 FP32
  accumulators). One 8-value A fragment and one 8-value B fragment then feed
  64 FMAs, cutting shared-memory traffic per FMA by 8×.
- **Interleaved thread tiles**: thread `tx` owns columns `tx, tx+16, tx+32,
  …`.  That makes the B-fragment shared reads consecutive across lanes
  (conflict-free) and keeps the global stores coalesced.  With contiguous
  per-thread columns the same kernel measures 0.871 TFLOP/s vs 1.036 here
  (4-way bank conflicts on the B reads).
- Shared tiles are padded (`As[128][17]`, `Bs[16][129]`).
- The tile loaders convert `half → float` on the way into shared memory,
  using scalar half loads so odd `K` (which would break float4 alignment)
  needs no separate path.  Edge tiles are bounds-checked and zero-filled; the
  epilogue applies `alpha`/`beta` and converts back to half.

Sweep (valid configurations only, all producing correct results):

| Variant | Time | TFLOP/s |
|---|---:|---:|
| **128×128, BK=16, 8×8, 16×16 (shipped)** | **2.07 ms** | **1.04** |
| 128×128, BK=16, 8×8, contiguous columns | 2.47 ms | 0.87 |
| 128×128, BK=32 | 2.10 ms | 1.02 |
| 128×128, BK=8 | 2.25 ms | 0.95 |
| 256×128, BK=16, 8×8, 16×32 (512 thr) | 2.22 ms | 0.97 |
| 192×128, BK=16, 8×8 | 2.42 ms | 0.89 |
| 128×192, BK=16, 8×8 | 2.39 ms | 0.90 |
| 64×64, BK=16, 4×4 | 2.29 ms | 0.94 |
| 128×128, BK=16, 4×4, 32×32 (1024 thr) | 3.36 ms | 0.64 |
| 64×64, BK=16, 2×2, 32×32 | 5.97 ms | 0.36 |

Larger tiles (256 rows / 1024 threads) do not fit the register budget: 64
accumulators plus fragments is ~128 registers/thread, so 256 threads per
block is already one block per SM.  BK = 8/16/32 differ by only a few
percent and their ordering flips between sessions (a separate sweep run
measured 1.93/2.07/1.86 ms for BK 8/16/32); 16 is kept as the middle choice.

### Naive baseline

`gemm_naive.cu` is the plainest correct tiled form — the starter's structure:
32×32 shared tiles, one output per thread, two shared loads per FMA.  It
measures **8.23 ms (0.261 TFLOP/s)**, 4.0× slower than the shipped kernel.
The naive form is the starting point every GEMM optimization improves on.

## Implementation — Numba

Same 32×32-tile structure written in Numba.  The 8×8 register tile is not
expressible there (no per-thread register array; a 64-element local array
would land in local memory), so this port is the starter's shape.  Numba also
cannot index `float16` arrays in CUDA kernels, so the matrices are passed as
`int16` views and converted with `libdevice.half2float` /
`float2half_rn` — the same bit patterns, so it is still a true
FP16-in/FP32-accumulate/FP16-out kernel.

It measures **13.10 ms (0.164 TFLOP/s)** — 6.3× behind the C++ kernel and
slower even than the C++ naive tile, since Numba does not unroll the inner
loop.  Kernels use `cache=True`.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, 6 SMs) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Times are medians after a warm-up, including `cudaDeviceSynchronize()`;
run-to-run variation is about ±1.5%.

## Correctness — CUDA C++

Compared against a float64 CPU reference rounded to half, with a tolerance of
`0.02 + 0.01·max|ref|` (FP16 output rounding dominates):

| Case | Result |
|---|---|
| challenge example (2×2×3, alpha=1, beta=0) | exact (`worst 0`) |
| 16³, 17×19×23, 32³, 33×65×17, 64³, 100×75×50, 128³, 257×131×97 | worst ≤ 0.031 |
| each shape also with alpha=0.5, beta=0.5 | worst ≤ 0.031 |
| benchmark shape 1024³ | worst 0.125 |

All 18 cases pass with zero elements outside tolerance.

## Correctness — Numba CUDA

All 17 cases pass (worst ≤ 7.8e-03 on the small shapes, 6.3e-02 at 1024³).

## Performance — 1024 × 1024 × 1024

| Implementation | Time | TFLOP/s | vs optimized |
|---|---:|---:|---:|
| starter (32×32 tile, 1 output/thread) | 7.72 ms | 0.278 | 3.7× slower |
| `gemm_naive.cu` (same shape) | 8.23 ms | 0.261 | 4.0× slower |
| **`gemm.cu`** (128×128 tile, 8×8 register block) | **2.07 ms** | **1.04** | 1.0× |
| Numba CUDA (32×32 tile) | 13.10 ms | 0.164 | 6.3× slower |
| *FP32 pipeline peak (no tensor cores on sm_61)* | — | *~2.27* | *2.2× above* |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test gemm.cu test_harness.cu
```

Numba:

```bash
python3 gemm.py
```

The script runs the correctness cases and the performance case.
