# Scatter-Add (row-wise) — CUDA C++ and Numba Results

Challenge: `output[m][d] = Σ src[i][d]` over all `i` with `index[i] == m`,
where `src` is `N×D`, `index` is length `N`, and `output` is `M×D` with
undefined initial contents (so every row must be written; rows no source maps
to are zero).  `N ≤ 2,000,000`, `D ≤ 1,024`, `M ≤ 1,000,000`; performance is
measured at `N = 1,000,000`, `D = 128`, `M = 50,000`.

Files:

- `scatter_add.cu` — CUDA C++ implementation of
  `solve(src, index, output, N, D, M)` (renamed from `starter.cu`).
- `scatter_add_naive.cu` — the naive baseline: one thread per element with a
  global `atomicAdd`, after zeroing the output.
- `scatter_add.py` — Numba CUDA port with the same signature.

## The starter is broken at the benchmark shape

Two problems:

1. Its launch grid is `ceil(N/8)` in `grid.y`, and `gridDim.y` is limited to
   65,535 — at `N = 1,000,000` that is 125,000, so the kernel never launches.
   (It works only for `N ≤ 524,280`.)  `solve` does not check the launch
   error, so the output is left at whatever the memset wrote.
2. Even when it does launch, it performs one global `atomicAdd` per element:
   128 million atomics at the benchmark shape.

Its measured 0.32 ms at the benchmark shape is just the `cudaMemsetAsync` of
the 25.6 MB output — the kernel never ran.  With the grid replaced by a 1-D
grid-stride loop it is correct and measures **19.7 ms**; that fixed form is
the naive baseline here.

## Where the time goes at N = 1,000,000, D = 128, M = 50,000

The streamed data is 512 MB of `src` + 25.6 MB of output (+ 4 MB of index and
4 MB of the sorted row list), so the memory-system floor is ~6.6–6.8 ms at
this GPU's ~82 GB/s read+write rate.

| Implementation | Time | Effective GB/s |
|---|---:|---:|
| `scatter_add_naive.cu` (128 M atomicAdds) | 19.7–19.8 ms | 27.4 |
| **shipped (counting sort, no output atomics)** | **8.63–8.65 ms** | **62.7** |
| *of which: sort phase (count + scan + scatter)* | *1.42 ms* | — |
| *of which: gather phase* | *~7.2 ms (75 GB/s)* | — |

**2.28×** over the atomic baseline.  The gather reads source rows in
destination-sorted order — random 512-byte chunks, coalesced within each row
— which reaches ~75 GB/s against the 82 GB/s copy ceiling; the sort's two
passes of atomics (count and scatter) and the 4 MB permutation list account
for the rest.

## Implementation — CUDA C++

1. `count_kernel` — `counts[m]++` for each source row (atomics on a 50,000
   entry array).
2. `scan_blocks_kernel` + `scan_sums_kernel` — exclusive prefix sum of the
   counts, chunked at 1024 per block with a second kernel scanning the block
   totals (chunk prefix added on the fly by later kernels).
3. `scatter_kernel` — `sorted_rows[pos] = i` with `pos` from an atomic cursor
   over a copy of the local prefix, plus the chunk prefix.
4. `gather_vec_kernel` — four destination rows per 128-thread block, one warp
   per row; each warp sums its rows' `float4`s and writes the output row once
   (no atomics on the output).  Rows with no sources are written as zero.  A
   scalar warp-per-row kernel handles `D % 4 != 0` or misaligned pointers.

The workspace (`counts`, `local`, `cursor`, `sorted_rows`, scan scratch) is
allocated once and cached across calls.

Alternative shapes were measured: a scalar gather (9.06 ms), a per-row 32-thread
block gather (8.74 ms), and the warp-per-row version here (8.65 ms).

### Naive baseline

`scatter_add_naive.cu` is the plainest correct form: zero the output, then one
thread per `(row, feature)` element with one `atomicAdd`.  It measures
**19.7 ms** — 2.28× slower — because all 128 M adds contend on 25.6 MB of
output through the atomic units.

## Implementation — Numba

The atomic form (zero kernel + `cuda.atomic.add`), since the counting-sort
pipeline needs chunked shared-memory scans and several kernels that are not
practical in Numba.  It measures **21.8 ms (24.8 GB/s)**, in line with the
C++ atomic baseline (Numba's atomics are a little slower).  Kernels use
`cache=True`.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, 6 SMs) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Times are medians after a warm-up, including `cudaDeviceSynchronize()`; the
C++ variants were compiled against the same harness and run interleaved.

## Correctness — CUDA C++

Checked against a float64 CPU reference with per-element tolerance
`2e-2 + 1e-4·|ref|`.  (The absolute term matters for the stress case that
sends all 50,000 rows to one destination: summing 50,000 float32 terms of
magnitude ~10 has ~0.02 of inherent float32 rounding error, which no
implementation can remove.)

| Case | Result |
|---|---|
| challenge example (4×2 → 3 rows) | exact |
| 1×1→1, 10×3→2, 1000×33→100, 5000×128→50 | max_abs ≤ 8e-05 |
| 100,000×17→1000, 100,000×256→100,000 | max_abs ≤ 8e-05 |
| all sources → one destination (worst-case contention) | max_abs 1.1e-02 |
| 1000 sources spread over 100,000 destinations (zeros written) | max_abs 1e-06 |
| benchmark shape, N = 1M, D = 128, M = 50,000 | max_abs ≤ 1.4e-02 (sums of ~20 rows) |

All 10 harness cases pass.

## Correctness — Numba CUDA

All 8 cases pass with the same tolerance.

## Performance — N = 1,000,000, D = 128, M = 50,000

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| starter (as shipped: never launches at this size) | 0.32 ms | — | invalid |
| `scatter_add_naive.cu` (atomic per element, fixed grid) | 19.7 ms | 27.4 | 2.28× slower |
| **`scatter_add.cu`** (counting sort + warp gather) | **8.65 ms** | **62.7** | 1.0× |
| Numba CUDA (atomic form) | 21.8 ms | 24.8 | 2.52× slower |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test scatter_add.cu test_harness.cu
```

Numba:

```bash
python3 scatter_add.py
```

The script runs the correctness cases and the performance case.
