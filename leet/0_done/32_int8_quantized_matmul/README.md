# INT8 Quantized Matrix Multiplication — CUDA C++ and Numba Results

Challenge: quantized GEMM
`C = clamp(rint(Σ_k (A−za)(B−zb)·sa·sb / sc) + zc, −128, 127)` with int8
inputs, int32 accumulation, round-half-to-even.  Benchmark:
**M = 8192, N = 4096, K = 2048** — 6.87e10 int8 multiply-adds.

Files:

- `int8_quantized_matmul.cu` — CUDA C++ implementation (renamed from
  `starter.cu`).
- `int8_quantized_matmul_naive.cu` — the naive baseline: the starter's dp4a
  GEMM verbatim **with one fix** (see below).
- `int8_quantized_matmul.py` — Numba CUDA port.

## Results at the benchmark shape

Same-process interleaved medians:

| Variant | Time | int8-MAC/s |
|---|---:|---:|
| **naive = starter (row-sum bug fixed)** | 269.2 ms | 255 G |
| **shipped** | **69.0 ms** | **996 G** |
| Numba CUDA port | 457.7 ms | 150 G |

Speedup: **3.90×**.  A register microbenchmark puts this GPU's dp4a peak at
1.10 T dp4a/s (4.4 T int8-MAC/s), so the kernel runs at ~23% of the
instruction peak; the rest is shared-memory and tile traffic.

## The starter's bug

The starter is an already-optimized dp4a kernel, but `compute_row_sums`
performs int32 reinterpret-cast loads at `A_row + k` where
`A_row = A + row·K`.  That address is 4-byte aligned only when `K` is a
multiple of 4; for any other K the kernel faults with "misaligned address"
and **poisons the CUDA context**, so every subsequent launch in the process
fails.  The naive baseline keeps the starter's structure with scalar row-sum
loads; this is what makes the 269 ms baseline measurable at all.

## Implementation — CUDA C++

The shipped kernel keeps the starter's exact algorithm (dp4a with the exact
zero-point correction `dot − zB·row_sum − zA·col_sum + K·zA·zB` and `rintf`
in the epilogue) and changes the tiling:

- **256×128 block tile, 8×8 per thread, 512 threads**, BK=32: shared loads
  per dp4a drop from the starter's 0.5 to 0.25.
- Shared tiles are the transposed B (`Bs[col][k]`, so four consecutive k are
  one aligned int32 for dp4a) and A (`As[row][k]`), with a 4-byte pitch pad
  to keep the int32 fragment loads aligned.
- Row/column sums are precomputed with the same warp-reduction kernels and
  cached buffers.

Measured shapes (same process, interleaved, full solve):

| Shape | Time |
|---|---:|
| 64×64, 4×4, BK=16 (starter shape) | 195–268 ms |
| 128×128, 8×4, BK=32 | 82.2 ms |
| 128×128, 8×8, BK=32 (256 thr) | 70.5 ms |
| **256×128, 8×8, BK=32 (shipped)** | **68.0 ms** |
| 128×256, 8×8, BK=32 | 70.4 ms |
| 8×8 with BK=64 | 78–88 ms |

Correctness: both challenge examples, random cases `1x1x1`, `5x7x11`
(zero-points 7/−3/12), `17x33x19` (extreme zero-points −128/127/127),
`64x128x64`, `129x130x131`, `256x256x257`, `512x512x512`, and the benchmark
shape (sampled 4096 outputs), all **exactly equal** to an int64 reference
that applies the same float32 scaling and `rintf` epilogue.

## Numba port

Numba has no dp4a intrinsic, so the port computes the products with scalar
int8→int32 multiply-adds in a 64×64 tile with 4×4 per thread (four times the
arithmetic instructions of the dp4a version), keeping the same exact
zero-point correction, precomputed row/column sums and `rintf` epilogue.
The benchmark-shape reference is sampled (a full int64 numpy matmul would be
a naive 6.9e10-operation loop).  Measured 457.7 ms (150 G int8-MAC/s).
