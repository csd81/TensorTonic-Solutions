# Grouped GEMM — CUDA C++ and Numba Results

Challenge: `C[off[g]:off[g+1]] = A[off[g]:off[g+1]] · B[g]` for `G` groups
(`group_offsets` gives the row boundaries; groups may be empty).
Benchmark: **G = 8, M_total = 8192, K = 1024, N = 2048** — 34.4 GFLOP.

Files:

- `grouped_gemm.cu` — CUDA C++ implementation (renamed from `starter.cu`).
- `grouped_gemm_naive.cu` — the naive baseline: the starter's 16×16
  one-output-per-thread kernel **with its shared-B race fixed** (see below).
- `grouped_gemm.py` — Numba CUDA port.

## Results at the benchmark shape

Same-process interleaved medians:

| Variant | Time | TFLOP/s |
|---|---:|---:|
| starter as shipped | 120.3 ms *(correct only for 16-row-aligned groups)* | 0.286 |
| **naive (starter structure, race fixed)** | 393.7 ms | 0.087 |
| **shipped** | **46.0 ms** | **0.746** |
| Numba CUDA port | 107.6 ms | 0.319 |

Speedup of the shipped kernel over the corrected naive baseline: **8.55×**;
2.62× over the starter as shipped on this (aligned) benchmark shape.

## The starter's bug

The starter caches a B tile in shared memory, but each thread fills it from
*its own row's* group matrix (`B + g·K·N`).  When a block's rows span a group
boundary the threads race to write that shared tile, and every row in the
block then multiplies against whichever group's weights landed there.
Verified with `G=2, offsets={0,2,4}`: the second group's rows produce
`A·B[0]` instead of `A·B[1]`.

It passes the benchmark only because equal 1024-row groups are multiples of
the 16-row block.  The naive baseline keeps the starter's shape but reads B
directly from the row's own group matrix, which is correct for any layout
(and, without B in shared memory, much slower — 394 ms).

## Implementation — CUDA C++

1. **Planning kernel** (1 thread): walks the groups and enumerates every
   (group, row-block) pair into lookup arrays `(blk_g, blk_row, blk_rows)`,
   plus the block count.  Each GEMM block then belongs to exactly one group,
   so no binary search or group check is needed in the hot loop.
2. **Grid**: `(ceil(N/BN), 1, max_blocks)` with
   `max_blocks = ceil(M_total/BM) + G`; blocks past the actual count return
   immediately, so no host-side synchronization is needed to size the grid.
3. **GEMM**: the register-blocked family (128×128 tile, 16×16 threads, 8×8
   per thread, BK=16, interleaved columns), with A/C indexed relative to the
   block's first row and B offset by the group's weight matrix.  Rows past
   the group's end are zero-filled on load and skipped on store, so a group
   whose row count is not a multiple of the tile still lands correctly.

Tile sweep at the benchmark shape (single runs, same harness):

| Shape | Time |
|---|---:|
| 256×128, 8×8 (16×32 thr) | 44.3 ms |
| 128×256, 8×8 (32×16 thr) | 45.1 ms |
| **128×128, 8×8 (16×16 thr, shipped)** | **45.9 ms** |
| 64×64, 4×4 (16×16 thr) | 50.0 ms |

Correctness: the challenge example, `1x1x1`, `17 rows, empty first group`,
empty middle and tail groups, uneven groups (`{0,1,2,100,101,300,301,302,800}`),
and the benchmark shape, all against a float64 per-group reference
(tolerance `1e-3 + 1e-4·|ref|`).

## Numba port

Same planning-kernel + group-aligned register-blocked GEMM structure
(64×64 tile, 16×16 threads, 4×4 per thread with 16 scalar accumulators,
int32 index arithmetic), with the planner and lookup arrays as cached device
arrays and the grid sized to the upper bound.  Measured 107.6 ms
(0.319 TFLOP/s).
