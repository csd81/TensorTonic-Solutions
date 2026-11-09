# Sparse × Dense Matrix Multiplication — CUDA C++ and Numba Results

Challenge: `C[M,K] = A[M,N] · B[N,K]`.  `A` is stored as a dense row-major
array but is 60–70% zeros (`nnz` is given, the positions are not).  Benchmark:
**M = 4096, N = 2048, K = 512** — 4.29e9 multiply-adds (8.6 GFLOP
dense-equivalent) over 32 MB of A, 4 MB of B and 8 MB of C.

Files:

- `sparse_matrix_dense_matrix_multiplication.cu` — CUDA C++ implementation of
  `solve(A, B, C, M, N, K, nnz)` (renamed from `starter.cu`).
- `sparse_matrix_dense_matrix_multiplication_naive.cu` — the naive baseline:
  the starter's 32×32 tiled kernel verbatim (one output per thread, two
  shared loads per multiply-add, sparsity ignored).
- `sparse_matrix_dense_matrix_multiplication.py` — Numba CUDA port.

## Results at the benchmark shape

Same-process interleaved medians:

| Variant | Time | Dense-equivalent TFLOP/s |
|---|---:|---:|
| **naive = starter kernel** | 27.95 ms | 0.307 |
| **shipped** | **7.29 ms** | **1.18** |
| Numba CUDA port | 23.9 ms | 0.36 |

Speedup of the shipped kernel over the starter: **3.8×**.

Correctness: the challenge example (`A 3×4`, `B 4×2` → `[[9,12],[9,12],[20,24]]`),
random cases `1x1x1`, `5x3x7`, `17x33x19`, `64x128x64`, `100x257x129`,
`256x512x512`, `129x130x131` (checked element-for-element in double), and the
benchmark shape (4096 sampled outputs; a full CPU double check would be 4.3e9
multiply-adds).  Tolerance `1e-3 + 1e-4·|ref|`.

On the benchmark shape, a *full* comparison against a float64 reference
shows 8 of 2,097,152 outputs outside a tighter tolerance: those outputs have
small true values (|ref| ≈ 0.4–6.8) while the accumulated terms are ~100 in
magnitude, so any float32 accumulation order leaves ~1.5e-3 of absolute error
there.  numpy's own float32 product differs from float64 by the same amount
at the same 8 outputs, so this is float32 arithmetic, not a kernel defect
(the 75 Numba port's self-check therefore uses a `1e-2 + 1e-4·|ref|`
tolerance).

## Implementation — CUDA C++

The register-blocked GEMM shape that wins the GEMM challenges on this GPU:
**128×128 block tile, 16×16 threads, 8×8 per thread (64 FP32 accumulators),
BK=16, interleaved per-thread columns** (consecutive lanes read consecutive
shared elements, so shared reads are conflict-free).  Shared loads per
multiply-add drop from 2 (starter) to 0.25, and tile loads are coalesced
along N (A) and K (B) with bounds-checked zero fill.

Tile variants measured at this shape (same process, interleaved):

| Variant | Time |
|---|---:|
| 128×128 tile, 16×16 thr, 8×8 (shipped) | 6.82 ms |
| 128×256 tile, 32×16 thr, 8×8 | 7.69 ms |
| 256×128 tile, 16×32 thr, 8×8 | 7.72 ms |
| 128×128 + per-row zero-skip | 7.88 ms |

## Why the sparsity does not help here

The nonzeros are random, which defeats every cheap way to skip work:

- **Block-level skipping**: a 128-row × 16-column A tile holds 2048 values;
  the probability that all are zero is `0.65^2048 ≈ 0`.
- **Warp-uniform skipping**: `ballot(a != 0)` over 32 lanes is essentially
  always nonzero (`1 − 0.65^32 ≈ 1`), so a warp never skips an FMA group.
- **Per-element skipping**: `if (a_frag[i] != 0)` compiles to *predicated*
  FMAs; predicated-off instructions still occupy issue slots.  Measured
  7.88 ms — 16% *slower* than the dense loop, because the compare/select per
  fragment row costs more than the skipped FMAs save.
- **CSR + SpMM**: building CSR costs a counting pass and a scatter over the
  32 MB of A (~1.5 ms), after which row-per-warp SpMM does `nnz·K = 1.5e9`
  multiply-adds instead of 4.3e9.  But every such formulation needs one
  shared-memory B load per multiply-add, which is ~5.6 ms of shared bandwidth
  on this GPU (6 SMs × 32 floats/cycle × 1.4 GHz) before any other cost —
  more than the dense kernel spends in total.  The dense GEMM's win is its
  register tile, which reuses each B fragment across 8 rows; sparsity cannot
  increase that reuse.

So the dense register-blocked kernel is the right answer for this input
distribution: the sparsity does not reduce the data that must be read, and it
cannot be made to reduce instructions without divergence.

## Numba port

64×64 block tile, 16×16 threads, 4×4 per thread with 16 explicit scalar
accumulators (shared loads per multiply-add 0.5 instead of the starter's 2),
all block/index arithmetic cast to int32.  The port cannot reach the C++
kernel's 8×8 register tile (Numba has no register-array abstraction that
keeps 64 accumulators out of local memory), and its generated address math is
dominated by 64-bit integer operations — a 1-D flat-shared-array variant
measured *slower* (35 ms) than the 2-D shared arrays (23.9 ms), so the 2-D
form is kept.

Measured 23.9 ms (0.36 TFLOP/s), 3.3× slower than the C++ kernel.
