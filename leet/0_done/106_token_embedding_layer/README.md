# Token Embedding + Layer Norm — CUDA C++ and Numba Results

Challenge: for every `(b, t)`,
`s = E_T[token_ids[b,t]] + E_P[position_ids[t]]`, then
`y = γ·(s − μ)/√(σ² + ε) + β` over the embedding dimension.  Benchmark:
**B = 32, T = 512, V = 30,000, P = 2,048, D = 768** — 16,384 rows; two row
reads + one row write of D floats = 150 MB, so the floor is ~1.8 ms at this
GPU's ~85 GB/s.

Files:

- `token_embedding_layer.cu` — CUDA C++ implementation (renamed from
  `starter.cu`).
- `token_embedding_layer_naive.cu` — the naive baseline: the starter's
  kernel verbatim (scalar loads, shared tree reduction, second pass over both
  rows).
- `token_embedding_layer.py` — Numba CUDA port.

## Results at the benchmark shape

Same-process interleaved medians:

| Variant | Time | Effective GB/s |
|---|---:|---:|
| **naive = starter kernel** | 2.170 ms | 69.6 |
| **shipped** | **1.813 ms** | **83.3** |
| Numba CUDA port | 3.026 ms | 49.9 |

Speedup: **1.20×**, and the shipped kernel is at the DRAM wall.

## Why the starter leaves ~20%

One block per `(b, t)` with 256 threads is the right shape for this problem,
but the kernel (a) uses scalar 4-byte loads, (b) reduces (sum, sumsq) through
a 9-step shared-memory tree, and (c) walks both embedding rows a second time
for the normalize pass — cheap for DRAM (the rows are in L1 by then) but not
free.

## Implementation — CUDA C++

1. **float4 rows**: each thread loads one 16-byte chunk of the token row and
   of the position row, forms `s` in registers and **keeps it there across
   the reduction** — with D ≤ 1024 every thread owns exactly one chunk, so
   the row is read once and written once.
2. **Warp-shuffle reduction** of `(sum, sumsq)` plus a shared-memory combine
   of 8 warp results (two syncthreads total).
3. Normalize from the registers, loading `γ`/`β` as float4.

Numerics: the variance is clamped at zero before `rsqrtf` (the one-pass
formula can round negative when the spread is tiny, and `rsqrtf` of a
negative argument is NaN), and the affine transform uses the challenge's
subtract form `γ·((s − μ)·inv_std) + β`.

A scalar fallback covers D not divisible by 4 (row bases would lose 16-byte
alignment) and re-reads the rows for the normalize pass.

Correctness: the challenge example, `1x1x1`, `1x1 D=4`, `2x3x5x4x17`,
`3x7x33x5x100`, `1x17x64x8x129`, `4x9 (D=1024)` and the benchmark shape, all
against a float64 reference (tolerance `1e-4 + 1e-4·|ref|`; max absolute
error 5.6e-6).

## Numba port

Same block-per-row structure with scalar loads, warp-shuffle + shared
reduction, the clamp and the subtract form.  Numba cannot keep the row in a
register across the reduction (no vector type; a local array would spill), so
the normalize pass re-reads the rows from L1 as the starter does.  Also note
both id arrays and embedding tables must be passed **flat** — a 2-D device
array indexed by `[i]` returns a row in Numba, not a scalar.  Measured
3.03 ms (49.9 GB/s).
