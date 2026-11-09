# Row-Wise Argmax — CUDA C++ and Numba Results

Challenge: `output[b] = argmax_v logits[b, v]`, ties resolved to the smallest
index.  Benchmark: **B = 256, V = 128,000** — 131 MB of reads; the memory
wall is ~1.6 ms at this GPU's ~85 GB/s.

Files:

- `row_argmax.cu` — CUDA C++ implementation (renamed from `starter.cu`).
- `row_argmax_naive.cu` — the naive baseline: the starter's kernel verbatim.
- `row_argmax.py` — Numba CUDA port.

## Results at the benchmark shape

Same-process interleaved medians:

| Variant | Time | Effective GB/s |
|---|---:|---:|
| **naive = starter kernel** | 1.672 ms (1.68 standalone) | 78.4 |
| **shipped** | **1.529 ms** | **85.7** |
| Numba CUDA port | 1.636 ms | 80.1 |

Speedup: **1.09×** — the starter was already essentially at the memory wall;
this challenge is a "starter is nearly optimal" case, and the README says so
rather than claiming a large win.

## What the shipped kernel changes

- **float4 loads** (4 consecutive logits per instruction), folded into the
  local (value, index) pair in ascending index order so the first-occurrence
  tie rule holds inside a chunk as well.
- The same merge rule (strictly greater, or equal with a smaller index) in
  the warp-shuffle and shared-memory block reductions.
- A scalar fallback for `V % 4 != 0` or unaligned base pointers.

Correctness: the challenge example, `1x1`, `1x7`, `5x33`, `17x1000`, `2x1e5`,
rows with several equal maxima (three ties per row), all-equal rows, rows
that are entirely `-3e38`, and the benchmark shape — all compared exactly
against a first-max reference (including the tie cases, where a
last-max implementation would fail).

## Numba port

Same block-per-row structure with scalar loads (Numba has no float4), warp
shuffles on the (value, index) pair and a shared array of warp results; the
scalar path means it measures 1.64 ms instead of 1.53.  Note the shared-array
shape must be a module-level literal (`NWARP`), not an expression like
`TPB // 32`.
