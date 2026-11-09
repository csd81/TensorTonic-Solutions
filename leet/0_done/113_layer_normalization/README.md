# Layer Normalization — CUDA C++ and Numba Results

Challenge: layer norm for `[N, C]` tensors — per-row mean/variance over the
`C` features, then `y = weight[j]·(x − μ)/√(σ² + ε) + bias[j]`.  Benchmark:
**N = 65,536, C = 512** — 134 MB in + 134 MB out; one read and one write is
268 MB, ~3.2 ms at this GPU's ~85 GB/s.

Files:

- `layer_normalization.cu` — CUDA C++ implementation (renamed from
  `starter.cu`).
- `layer_normalization_naive.cu` — the naive baseline: the starter's kernel
  verbatim (three scalar passes over each row).
- `layer_normalization.py` — Numba CUDA port.

## Results at the benchmark shape

Same-process interleaved medians:

| Variant | Time | Effective GB/s (2×traffic) |
|---|---:|---:|
| **naive = starter kernel** | 7.62 ms (7.21 standalone) | 44 |
| **shipped** | **3.28 ms** | **82** |
| Numba CUDA port | 8.88 ms | 30 |

Speedup: **2.33×** — the shipped kernel reads each row once and writes it
once (the starter reads each row three times), and it runs at the wall.

## Why the starter leaves 2× on the table

One block per row is already the right shape, but the kernel makes three
passes over every row: mean, variance, then normalize — 536 MB of traffic for
a problem that needs 268 MB — and uses scalar loads.

## Implementation — CUDA C++

1. One block per row; the block size is chosen so each thread owns exactly one
   `float4` chunk (`C/4 ≤ 1024` for `C ≤ 4096`), loaded **once** into
   registers.
2. Two block reductions over those registers: `sum → mean`, then
   `Σ(x − mean)² → variance`.  Deriving the variance from the register values
   (rather than `sumsq − mean²`) keeps it accurate when the row mean is large
   relative to its spread.
3. Normalize and write from the same registers (weight/bias as `float4`), so
   the row is never re-read.

A scalar path covers `C % 4 != 0` (register-resident up to C = 1024,
re-reading L1 beyond that).

Two correctness traps were found while validating:

- **A race between the two reductions**: both reuse one shared slot, and the
  second reduction's warp leaders could overwrite the broadcast value before
  slower threads read the first result — 44 K of 33.5 M outputs were wrong
  before the barrier after the read was added.  The broadcast helper now
  ends with a `__syncthreads()`.
- Inactive threads (`threadIdx.x ≥ C/4`) must contribute **zero** to the
  variance; their zeroed registers otherwise add `4·mean²` to the reduction.

Correctness: the challenge example, `1x1`, `1x7`, `5x3`, `17x33`, `1000x64`,
`3x1024`, `2x4096` (max C), `2x1023` (odd C) and the benchmark shape (max
absolute error 4.3e-6).

## Numba port

Same register-resident structure where the block covers the row in one
element per thread, warp-shuffle + shared reductions with the post-read
barrier, and a strided fallback for larger C.  Scalar loads, so it measures
8.88 ms.
