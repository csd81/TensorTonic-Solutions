# Top-k Selection — CUDA C++ Results

Challenge: given `N` 32-bit floats, write the `k` largest to `output` in
descending order. Benchmark: **N = 50,000,000, k = 100** — 200 MB of reads
(the output is 400 B), so a pure read of the same buffer is the floor.

Measured on this machine's Quadro P2000 (6 SMs, 96 KB shared/SM, 1.6 GHz):
a float4 read-only kernel over the same 200 MB takes **2.30 ms (87 GB/s)**.

Files:

- `top_k_selection.cu` — CUDA C++ implementation of `solve(input, output,
  N, k)` (the shipped kernel).
- `top_k_selection_naive.cu` — the challenge starter's kernel verbatim,
  kept as the reference implementation it replaces.
- No Numba CUDA port in this pass.

## Results at N = 50,000,000, k = 100

| Variant | Time | Effective GB/s |
|---|---:|---:|
| read floor (float4 sum over the same 200 MB) | 2.30 ms | 87 |
| **shipped** | **2.72 ms median (2.69 min)** | **73** |

Three independent 21-round runs gave 2.71–2.83 ms; the spread is clock and
desktop-compositor noise. The kernel is within 18% of the pure-read floor,
i.e. the selection work is almost entirely hidden behind the required memory
traffic.

The starter cannot be timed at this shape: its fast path launches
`BLOCK_SIZE * k * 4 = 100 KB` of dynamic shared memory without the >48 KB
opt-in, so the launch fails with `invalid argument` at k = 100 (verified) and
writes nothing. Its only other path, the full bitonic sort of all N values,
is gated behind k > 256. At the benchmark shape the starter has no working
path.

## Correctness

All cases are compared element-wise with the exact values produced by
`std::partial_sort` (bitwise-equal float values; ties may be in any order).
37 cases, all exact:

| Group | Cases |
|---|---|
| challenge examples | both examples |
| tiny | N=1, N=2 (k=1 and k=2), N=3 all-equal, N=5 all-negative |
| random uniform | N=1k (k=1,10,100,1000), 100k (k=100), 1M (k=48,49,100,257,1024), 50M (benchmark shape), 100M |
| sorted | ascending/descending 100k (incl. all-positive and all-negative), ascending/descending 5M |
| duplicates | 7 distinct values 100k, 3 distinct 100k k=1000, 2 distinct 100k, 3 distinct 5M, all-equal 100k k=100, all-equal 50k k=1000, all-equal 5M k=1024 |
| special values | all `-FLT_MAX` 10k, `FLT_MAX`/`-FLT_MAX` mixed 100k, `+inf`/`-inf` mixed 100k |
| k boundaries | k=1, 48 (starter's launch limit), 49, 100, 257, 1000, 1024, fallback k=1025 (N=5k and 100k), k=N=100k |

N=100M (the challenge maximum) was additionally checked against the CPU
reference. NaN is not treated specially (comparisons with NaN are false); the
reference used here is finite values plus ±inf.

## Implementation

One pass over the data with a per-block threshold buffer, then a tiny second
pass over the blocks' candidates.

1. **Per-block candidate buffer.** Each block keeps a `CAP = 2048` float
   shared buffer, a threshold `thr`, and a count `cnt`, all in shared memory.
   The buffer starts filled with `-inf` and `thr = -inf`. A scanned value is
   appended (shared `atomicAdd` slot) iff it is greater than `thr`.
2. **Uniform capacity check.** The scan advances `CHUNK = 4` grid strides per
   phase. When the buffer passes `CAP - CHUNK*TPB = 1024` entries, the block
   bitonic-sorts the buffer descending, keeps the top k, and sets `thr` to
   the k-th largest. Since `thr` is always a valid lower bound on the block's
   final k-th largest, no element that could belong to the top k is ever
   dropped; all-equal and duplicate-heavy inputs work by construction. At
   most 1024 values are appended between checks, so the buffer can never
   overflow.
3. **Prefetch, then append.** The phase's four loads go into registers first
   (`xs[c] = i < n ? input[i] : -inf`), and only then is the append decided.
   Independent loads issued together are what keep the kernel near the memory
   wall; this one change took the benchmark from 3.17 ms to 2.78 ms. The
   earlier form (`load; if (x > thr) append` per element) serialized the
   loads behind the branch.
4. **One occupancy wave.** The grid is `8 * SM_count` blocks (48 here), capped
   by `ceil(N/TPB)`. With `CAP = 2048` (8 KB of shared memory) and 256
   threads, 8 blocks fit per SM, so this is exactly one full wave. Block count
   matters a lot because every block sorts and compacts its own buffer; with
   the prefetch kernel: 48 blocks 2.76 ms, 44 → 2.80, 40 → 2.86, 52 → 3.39,
   60 → 3.24, 96 → 3.03. Using thousands of blocks is much worse (1024 blocks
   measured 9.4 ms before the prefetch change, since the per-block sort
   dominates).
5. **Emit and final reduction.** Each block writes its k values; a second
   launch of the same kernel with one block reduces the `blocks*k = 4800`
   candidates to the final sorted top-k.

Two correctness traps that the first version of this file fell into, both
fixed here:

- **Uninitialized shared buffer.** Sorting an uninitialized buffer at
  compaction/final time lets garbage outrank real values; the buffer must
  start at `-inf`. This produced whole-array garbage for all-negative and
  all-equal inputs.
- **Non-uniform compaction.** A plain `__syncthreads(); if (cnt > T) { sort }`
  has a window in which a fast warp reads `cnt` and enters the sort while a
  slow warp is still appending the previous phase's values — two barriers in
  flight, a data race, and dropped candidates (observed: 13/100 wrong at the
  benchmark shape). `__syncthreads_or(cnt > T)` makes both the phase barrier
  and the decision: every thread gets the same answer, the last appending
  thread observes the final count, and the sort runs only after all appends
  are complete. This costs one barrier per phase and is race-free.

Bitonic sort cost per compaction is 66 barrier steps for 2048 values (12
stages); for uniform random data a block compacts once, so compactions are
not the bottleneck. For adversarially sorted input the threshold rises
slowly and a block compacts every few phases; that is correct but slower
(the benchmark distribution is random).

The large-k path (k > 1024, not benchmarked) is the starter's approach: map
floats to sortable uints, pad to a power of two with 0, run the full bitonic
sort, extract k. Its `bitonic_sort_step` now bounds-checks the thread index.
`k <= 1024` is the fast path.

## Build / run

The file exports `solve`; compile it with a harness that provides `main`:

```bash
nvcc -O3 -arch=sm_61 -o test top_k_selection.cu test_harness.cu
```

The benchmark harness allocates N=50,000,000 random uniforms in [-1, 1),
warms up 5 times, then times 21 runs with CUDA events and reports the median;
a float4 read-only kernel over the same buffer gives the 2.30 ms floor above.
