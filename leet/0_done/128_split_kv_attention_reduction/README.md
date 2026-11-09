# Split-KV Attention Reduction — CUDA C++ and Numba Results

Challenge: merge Flash-Decoding partial attention results with the
log-sum-exp trick.  Given `partial_out (S, H, D)` and `partial_lse (S, H)`,

```
m_h = max_s lse[s,h]
w_s = exp(lse[s,h] - m_h)
out[h,d] = Σ_s w_s · partial_out[s,h,d] / Σ_s w_s
```

Benchmark: **S = 128, H = 64, D = 128** — 4.1 MB read (4 MB of partial
outputs), ~1 M multiply-adds, so this is a small, bandwidth-bound kernel.

Files:

- `split_kv_attention_reduction.cu` — CUDA C++ implementation of
  `solve(partial_out, partial_lse, output, S, H, D)` (renamed from
  `starter.cu`).
- `split_kv_attention_reduction_naive.cu` — the naive baseline: the starter's
  kernel verbatim (one block per head, one thread per dimension, every thread
  re-reads the lse column and re-evaluates `exp()` per split).
- `split_kv_attention_reduction.py` — Numba CUDA port.

## Results at S = 128, H = 64, D = 128

Same-process interleaved medians (200 rounds):

| Variant | Time | Effective GB/s |
|---|---:|---:|
| **naive = starter kernel** | 68.1 µs | 62.6 |
| **shipped** | **58.0 µs** | **73.4** |
| Numba CUDA port | 135 µs | 31.1 |

Speedup of the shipped kernel over the starter: **1.17×**.  The problem is
memory-bound: 4.1 MB at this GPU's ~80 GB/s is ~52 µs, so the shipped kernel
is within ~12% of the read floor and there is little room left.

Correctness: the challenge example (`S=2, H=2, D=2` — heads weight their
splits ½:½ and ¾:¼, output `[[2,3],[3,2]]`) plus `1x1x1`, `1x8x16`,
`5x3x7`, `17x33x19`, `64x8x128`, the benchmark shape, and the maximum shape
`256x128x256`, all against a float64 reference (tolerance `1e-4 + 1e-4·|ref|`,
max absolute error 3.6e-7).

## Why the starter is slow, and what changed

The starter is already close to the memory floor; its waste is redundant
compute and extra work per thread:

- Every thread (one per dimension) re-reads the whole `lse` column and
  evaluates `exp()` for every split: 8,192 threads × 128 splits = 1 M
  exponentials where only S·H = 8 K are needed.
- Scalar 4-byte loads: a block of 128 threads reads one 512-byte row per
  split, one instruction per 128 bytes.

The shipped kernel uses **one warp per head** (64 warps total):

1. The lse max and the stable weights are computed **once per (head, split)**
   with `__shfl_xor_sync` reductions over the warp and kept in shared memory.
2. Each lane owns **four consecutive dimensions** (`float4`), so a warp reads
   a 512-byte split row per instruction.
3. The split loop is unrolled by four, keeping four loads in flight per lane.
4. `D ≤ 256` allows up to two float4 chunks per lane, handled with two
   explicit accumulators; the chunk-count branch is hoisted outside the split
   loop (with it inside, the kernel measured 65.8 µs instead of 58.0 µs).
   A scalar path covers `D` not divisible by 4.

Variants measured at the benchmark shape (same process, interleaved):

| Variant | Time |
|---|---:|
| starter / naive (one block per head, scalar) | 68.1 µs |
| one warp per head, 1 split/iter, float4 | 63.2 µs |
| one warp per head, 4 warps per head, split range divided | 58.6 µs |
| one warp per head, 2 splits/iter, float4 | 59.4 µs |
| **one warp per head, 4 splits/iter, float4 (shipped)** | **58.0 µs** |

## Numba port

Same structure: one warp per head, `cuda.shfl_xor_sync` for the max and
weight-sum reductions, weights in a shared array, and each lane accumulating
its dimensions in registers with a two-way unrolled split loop.  Numba has no
`float4` type, so a lane reads its dimensions as scalars at a 32-lane stride
(one 128-byte transaction per warp instruction) — that, plus Numba's
codegen, is why the port measures 135 µs against the C++ kernel's 58 µs.
