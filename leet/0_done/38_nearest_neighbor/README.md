# Nearest Neighbor — CUDA C++ and Numba Results

Challenge: for each of `N` 3-D points, fill `indices[i]` with the index `j != i`
minimizing the squared Euclidean distance.  Points are floats in `[-1000, 1000]`,
`N ≤ 100,000`; performance is measured at `N = 10,000` (1e8 point pairs).

Files:

- `nearest_neighbor.cu` — CUDA C++ implementation of `solve(points, indices, N)`
  (renamed from `starter.cu`).
- `nearest_neighbor_naive.cu` — the naive baseline: the starter's kernel
  verbatim (AoS shared tile, one query per thread, self-test in the inner
  loop, 256-thread blocks).
- `nearest_neighbor.py` — Numba CUDA port with the same signature.

## Results at N = 10,000

Same-process interleaved medians (the two implementations launched
alternately in one process, 40 rounds; absolute numbers drift ~4% between
processes on this desktop GPU):

| Variant | Time | Throughput |
|---|---:|---:|
| **naive = starter kernel** | 1.54–1.57 ms | 63.6–64.8 Gpairs/s |
| **shipped** | **1.28–1.31 ms** | **76.6–78.0 Gpairs/s** |
| Numba CUDA port | 2.04 ms | 48.9 Gpairs/s |

Speedup of the shipped kernel over the starter: **1.20×**.  At the challenge's
`N` ceiling the advantage holds and grows slightly:

| N | shipped | naive (starter) | ratio |
|---:|---:|---:|---:|
| 10,000 | 1.2827 ms (78.0 Gpairs/s) | 1.5432 ms (64.8) | 1.20× |
| 100,000 | 122.49 ms (81.6 Gpairs/s) | 147.80 ms (67.7) | 1.21× |

Correctness: all cases in the harness pass — the challenge example
`(0,0,0), (1,0,0), (5,5,5) -> [1,0,1]`, `N = 1` (`-1`, no candidate), and
random `N = 2, 3, 10, 100, 1000, 5000, 10000`.  The harness accepts any index
whose distance is within `dref * (1 + 1e-4) + 1e-6` of the float64 brute-force
minimum, so ties (including duplicate points) are fine.

## Where the time goes

The kernel evaluates 1e8 pairs; each evaluation is ~3 substractions,
3 multiply-adds, one compare and a predicated index update.  Every thread in a
warp examines the *same* candidate index at the same time (the `k` loop is
uniform across the block), so the three shared-memory loads per candidate are
warp broadcasts, not per-lane reads.  Shared-bank layout therefore does not
matter: a same-process comparison of an AoS tile against the shipped SoA tile
gives 1.317 ms vs 1.306 ms (~1%).  The kernel is **instruction-throughput
bound**, roughly 10–12 instructions per pair ≈ 0.9–1.0 T instructions/s
against an SM issue ceiling of ~1.07 T lane-instructions/s (6 SMs × 128
lanes × 1.4 GHz) — about 85% of issue capacity.  Data volume is irrelevant:
the 120 KB point array fits in L2.

Optimization ladder (N = 10,000, interleaved medians in one process; the last
two rows come from a separate same-process run, so compare within a table,
not across):

| Variant | Time | Note |
|---|---:|---|
| starter: AoS tile, 1 query/thread, 256 thr | 1.5735 ms | baseline |
| + TQ=2 queries/thread + self-skip fast path, 128 thr | 1.3167 ms | AoS twin of shipped |
| + SoA tile (shipped) | 1.3061 ms | +1% |
| shipped kernel, 256 thr instead of 128 | 1.4698 ms | −12% |
| SoA tile, TQ=2, no self-skip, 128 thr | 1.3417 ms | self-skip worth 6.2% |
| SoA tile, TQ=2 + self-skip, 128 thr | 1.2584 ms | same run as row above |

Other variants tested and rejected: one query per thread with a per-iteration
validity guard (1.93 ms), three or four queries per thread (1.56–2.15 ms;
the grid shrinks to too few threads), four candidates per `float4` shared
load (1.37 ms; the compiler's scalar broadcasts already saturate the load
pipe), reading candidates directly from L2 without shared memory (1.60 ms;
the shared broadcast is cheaper), tile sizes of 128/512 (within 1% of 256),
and 64-thread blocks (1.36 ms).

## Implementation — CUDA C++

`solve` launches `ceil(N / 256)` blocks of 128 threads; each thread owns two
consecutive queries held in registers for the whole kernel.

- **Tiling**: points are staged into a 256-point shared tile, reloaded once
  per tile by any block quorum (`t += blockDim.x` stride, so any block size
  works).  Each candidate is read once from shared memory and used for both
  of the thread's queries.
- **Self-skip fast path**: a thread's own queries occupy `[q0, q0+1]`.  When
  the current tile does not intersect that range — 39 of 40 tiles at
  N = 10,000 — the inner loop omits the `j != q` test entirely.  Only the
  intersecting tile runs the guarded variant.
- **Tail handling**: threads with `q0 >= N` load nothing, compute on zeroed
  register queries (harmless), and skip their stores; `indices[]` of a
  lone point stays `-1`.
- **Precision**: squared distances are evaluated in FP32 (the comparisons are
  exact translations of the challenge definition); only the harness reference
  is FP64, with a 1e-4 relative tolerance to absorb rounding ties.

## Numba port

`nearest_neighbor.py` keeps the same structure: SoA shared tile
(`cuda.shared.array(256, float32)`), two queries per thread in registers, and
the tile-aware self-skip.  Two Numba-specific changes were needed:

- **int32 bookkeeping throughout**: with the tile counters typed `int64`,
  Numba's `range(int32)` promotion forced 64-bit loop indices and cost ~10%;
  declaring the tile counter and `tile_end` as `int32` keeps `k`, `j` and the
  stored index 32-bit.
- **Hoisted validity guards**: the C++ kernel's per-iteration `q0 < N` checks
  are dropped entirely; tail threads run with dummy zero queries and only
  their final stores are guarded.

Measured 2.04 ms at N = 10,000 (48.9 Gpairs/s), the best of block sizes
64/128/256/512 (2.11/2.04/2.29/2.17 ms).  The PTX shows no local-memory
spills.  The Numba port is ~1.6× slower than the C++ kernel.
