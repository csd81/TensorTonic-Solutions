# Group Normalization — CUDA C++ and Numba Results

Challenge: group norm for `(N, C, H, W)` tensors — per `(batch, group)` mean
and variance over the contiguous `(C/G)·H·W` block, then
`y = γ_c·(x − μ)/√(σ² + ε) + β_c`.  Benchmark: **N=8, C=512, H=64, W=64,
G=32** — 256 groups of 65,536 elements, 67 MB in and 67 MB out.

Files:

- `group_normalization.cu` — CUDA C++ implementation (renamed from
  `starter.cu`).
- `group_normalization_naive.cu` — the naive baseline: the starter's kernel
  verbatim.
- `group_normalization.py` — Numba CUDA port.

## Results at the benchmark shape

Same-process interleaved medians:

| Variant | Time | Effective GB/s |
|---|---:|---:|
| **naive = starter kernel** | 2.704 ms (2.66 standalone) | 74.5 |
| **shipped** | **2.470 ms** | **81.5** |
| Numba CUDA port | 2.687 ms | 74.9 |

Speedup: **1.09×**.  A 65,536-element group (256 KB) does not fit in shared
memory, so the kernel must read it twice (stats and normalize) plus write it
once: 201 MB, which is ~2.37 ms at the ~85 GB/s wall — the shipped kernel is
within 4% of that.

## Why the starter leaves ~10%

The starter's per-element index math recomputes `(channel, hw)` with runtime
divisions and recombines them into a flat index — unnecessary, because a
group is one contiguous range.  It also uses scalar 4-byte loads.

## Implementation — CUDA C++

1. The group's base offset is computed once per block, and both hot loops
   walk the contiguous range directly (the only division is per 16-byte
   chunk in the write pass).
2. The stats pass accumulates sum and sum-of-squares with eight FP32
   accumulators over float4 loads.  (An early version split them into two
   loops — that is a third read of the group and measured worse; the README
   notes it because it is an easy mistake to make.)
3. Warp-shuffle + shared reduction of `(sum, sumsq)`, variance clamped at
   zero, `rsqrtf`, and a float4 normalize pass with the per-chunk channel
   index.

A scalar fallback covers `H·W` not divisible by 4.

Correctness: both challenge examples, `1x1x1x1 G=1`, `G=C` (instance norm),
`G=1` (layer norm), `2x8x4x4 G=2`, `1x64x8x8 G=8`, `3x16x17x13 G=4` (odd
sizes), `1x1024x1x1 G=64` and the benchmark shape, against a float64
reference (max absolute error 3.8e-6).

## Numba port

Same contiguous-range structure with scalar loads, warp-shuffle + shared
reduction, clamped variance and subtract-form affine.  Measured 2.687 ms
(74.9 GB/s).
