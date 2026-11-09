# Matrix Power — CUDA C++ and Numba Results

Challenge: compute `output = A^P` for a dense `N × N` FP32 matrix by
exponentiation by squaring.  `1 ≤ N ≤ 1024`, `1 ≤ P ≤ 20`; performance is
measured at `N = 512`.

Files:

- `matrix_power.cu` — CUDA C++ implementation of `solve(input, output, N, P)`
  (renamed from `starter.cu`).
- `matrix_power_naive.cu` — the naive baseline: the starter's shape verbatim
  (32×32 shared tile, one output per thread, a `cudaMemcpy` after every
  product, `cudaMalloc`/`cudaFree` per call).
- `matrix_power.py` — Numba CUDA port with the same signature.

## Results at N = 512

Same-process interleaved medians:

| P | matmuls | naive = starter | shipped | speedup |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0.032 ms | 0.031–0.032 ms | 1.0× (pure copy) |
| 2 | 1 | 1.167 ms | 0.430–0.480 ms | 2.4–2.7× |
| 5 | 3 | 3.125 ms | 1.205 ms | 2.6× |
| 10 | 4 | 4.099 ms | 1.575 ms | 2.6× |
| 20 | 5 | 5.129 ms | 1.926–2.127 ms | 2.4–2.7× |

(Standalone runs give 0.430 / 1.205 / 1.575 / 1.926 ms for P = 2/5/10/20; the
same-process pair runs give 0.480 and 2.127 ms for P = 2 and 20.  The
benchmark's exponent is not stated by the challenge, so all values are
reported.)

Numba CUDA port: 1.29 ms at P=2 and 4.25 ms at P=20 (~2.2× slower than the
C++ kernel).

Correctness: both challenge examples (exact), `P=1` copy, and
`N = 1, 2, 8, 16, 33, 64, 100, 129, 256, 512` with `P = 2…20`, plus an
unscaled `64×64` case (`A^3` with entries up to 1e4), all match a float64
reference.  Tolerance is anchored to the reference's magnitude
(`1e-4·max|ref| + 1e-4·|ref|`): the iterated powers of scaled random
matrices decay to ~1e-7, so a floor-style absolute tolerance would let
unwritten or zeroed output rows pass — which is exactly how an earlier
inconsistent tile shape went unnoticed before the harness was tightened.

## Why the starter is slow

- 32×32 shared tile with **one output per thread**: two shared loads per
  multiply-add, ~0.3 TFLOP/s (0.87 ms per 512³ product; the kernel spends its
  time in the shared-memory pipe, not the FMA units).
- A `cudaMemcpy` of the full matrix after **every** product (seven 1 MB
  copies for P=20) plus a `cudaMalloc`/`cudaFree` pair per call.

## Implementation — CUDA C++

1. **GEMM**: the register-blocked family from the other matmul challenges,
   sized for N = 512: 64×128 block tile, 16×16 threads, 4×8 per thread with
   interleaved columns.  Shared loads per multiply-add are 0.375 (vs 2), which
   puts the kernel near the shared-memory roofline for 512³ products:
   0.42 ms measured versus a 0.37 ms load-bound estimate (0.62–0.70 TFLOP/s).
2. **Left-to-right binary exponentiation with the read-only input as the
   base**: `R = A`, then for each bit below the most significant one,
   `R = R²` and, if the bit is set, `R = R·A`.  Only two scratch buffers are
   needed and the result ping-pongs between them **by pointer swap**, so the
   whole solve performs exactly two 1 MB device copies (input → scratch,
   result → output) instead of the starter's seven.
3. **Cached workspace**: the two buffers are allocated once and reused across
   `solve` calls, so a repeated benchmark pays no allocation.

The matrix product count is minimal for the exponent (P=20 needs five
products: four squarings and one multiply), so the GEMM speed is the only
remaining lever.

GEMM shape sweep at N=512, P=20 (same-process interleaved, full solve):

| Shape (TM×TN, threads, tile) | Time |
|---|---:|
| **4×8, 16×16, 64×128 (shipped)** | **2.155 ms** |
| 4×4, 16×16, 64×64 | 2.184 ms |
| 4×8, 16×32, 128×128 (512 thr) | 2.421 ms |
| 4×4, 32×32, 128×128 (1024 thr) | 2.324 ms |
| 8×4, 16×16, 128×64 | 2.771 ms |
| 8×8, 16×16, 128×128 | 2.643 ms |

At N=512 the 8×8 tile (0.25 loads/FMA) has too few blocks (16) to hide
latency; the 4×8 shape wins by having twice the blocks with only 50% more
shared traffic.

## Numba port

Same structure with a Numba-friendly 64×64 tile and 4×4 per thread (16
explicit scalar accumulators), int32 block/index arithmetic, and the two
ping-pong buffers as cached device arrays swapped by reference in Python
(exact-size views are passed so a cached buffer larger than the current N can
be reused).  Measured 1.29 ms (P=2) and 4.25 ms (P=20), ~2.2× slower than the
C++ kernel.
