# Radix Sort — CUDA C++ and Numba Results

Challenge: sort `N` 32-bit unsigned integers in ascending order with a radix
sort.  `1 ≤ N ≤ 100,000,000`; performance is measured at `N = 50,000,000`.

Files:

- `radix_sort.cu` — CUDA C++ implementation of `solve(input, output, N)`
  (renamed from `starter.cu`).
- `radix_sort.py` — Numba CUDA port with the same signature.

## The starter — and its bug

The starter is a 4-bit LSD radix sort (8 passes): per-block histograms, a
recursive Blelloch exclusive scan of the `(16 × nblocks)` matrix (three
kernels plus `cudaMalloc`/`cudaFree` per recursion level), and a ballot-based
warp ranking in the scatter.  It measured **138.5 ms** at N = 50 M.

It is also **incorrect for values ≥ 2^28**.  With two ping-pong buffers and
8 passes, the sorted data ends up in `d_temp`:

```
pass 0 -> output, 1 -> d_temp, 2 -> output, ... 7 -> d_temp
```

so `output` holds the state after pass 6.  That is only the final answer when
pass 7 (the top nibble) changes nothing — i.e. when every value is below
2^28.  The challenge examples, small ranges and "already sorted 0..9999" all
pass; any full-range random array fails with the right multiset but the wrong
order.

## Implementation — CUDA C++

The final kernel uses **8-bit digits (4 passes instead of 8)** and the classic
three-stage structure per pass.

`agg[d * nblocks + b]` holds the number of elements of digit `d` in block `b`.
An exclusive scan over that array **in memory order** gives, for block `b` and
digit `d`, exactly the destination base:

```
base(d, b) = sum over digits d' < d of total[d']        (all blocks)
           + sum over blocks b' < b of count_b'[d]
```

The first term is a *global* quantity — it includes blocks that have not run
yet — so unlike a prefix sum, this cannot be resolved with a cross-block
decoupled lookback: every block's histogram must exist before any offset is
known.  Hence three stages per pass:

| Stage | Work | Traffic |
|---|---|---|
| `hist_kernel` | per-block digit histogram, `uint4` loads, shared atomics, written digit-major | 200 MB read |
| `scan` | exclusive scan of the `(256 × nblocks)` matrix (per-chunk scan + one-block scan of the chunk totals; the chunk prefix is folded into the scatter's offset preload) | 25 MB × 2 |
| `scatter_kernel` | stable block-local rank, then a staged write | 200 MB read + 200 MB write |

The ranking uses the pre-Volta `MatchAny` trick — `RADIX_BITS` ballots per
warp item give each lane the mask of lanes sharing its digit — plus the
running per-warp histogram for the count over the warp's earlier items, and
the block digit prefix (a 256-wide shared scan) for the count over earlier
warps.  Elements are then staged in shared memory at their block-local sorted
position and written out in that order, so consecutive global stores belong to
the same digit run and stay coalesced.  The per-warp histograms are 16-bit
(4 KB for 8 warps), which is what lets three blocks fit per SM; with 32-bit
histograms the kernel drops to two blocks/SM and loses ~11%.

The last pass writes straight into `output`, and all buffers are cached across
calls.

### Tuning

Median of 10 runs at N = 50 M:

| Configuration | Time |
|---|---:|
| ITEMS=4, 32-bit histograms | 81.7 ms |
| ITEMS=16, 32-bit histograms | 75.2 ms |
| ITEMS=8, 32-bit histograms | 63.4 ms |
| ITEMS=8, **16-bit histograms** (3 blocks/SM) | 53.0 ms |
| + fold the chunk prefix into the scatter | **49.1 ms** |

Scatter ablation at ITEMS=8 (one pass, 200 MB read + 200 MB write):

| Variant | Time |
|---|---:|
| pure copy of the same tile | 4.98 ms |
| + ranking + shared staging, contiguous writes | 6.28 ms |
| full scatter (digit runs of ~8 elements) | 10.29 ms |

So the ranking and staging cost ~1.3 ms per pass and the scattered stores
another ~4 ms.

## Implementation — Numba

The same three-stage 8-bit algorithm: per-block histograms with shared atomics,
the two-kernel matrix scan, and the ballot-based ranking (`cuda.ballot_sync`,
`cuda.popc`, `cuda.lanemask_lt`) with the same staged write.  Numba has no
16-bit shared arrays that index cheaply, so the per-warp histograms are 32-bit
(8 KB) here.

One Numba gotcha worth recording: its two-argument `min()` does not type in
CUDA kernels — it resolves to the one-argument builtin and fails with
"Signature mismatch: 2 argument types given, but function takes 1 arguments".
The port uses an explicit comparison instead.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, 6 SMs) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Kernel times are medians of 10 runs after a warm-up, with the inputs on the
device, including `cudaDeviceSynchronize()`.

## Correctness — CUDA C++

Every case is compared element-by-element against `std::sort`; all match
exactly:

| Case | N | Result |
|---|---:|---|
| challenge example 1 (`[170,45,75,90,2,802,24,66]`) | 8 | exact |
| challenge example 2 (`[1,4,1,3,555,1000,2]`) | 7 | exact |
| single element | 1 | exact |
| two equal / two reversed | 2 | exact |
| `{0, 2^32-1, 0, 2^32-1, 1, 2^31}` | 6 | exact |
| all equal | 1,000 | exact |
| already sorted / reverse sorted | 10,000 | exact |
| four-bit values only | 1,000 | exact |
| random | 1,000 / 100,000 / 1,000,000 | exact |

At the benchmark size the output is verified sorted and the multiset is
checked by a 64-bit sum; both hold.

## Correctness — Numba CUDA

All twelve cases match `np.sort` exactly, including the full-range random
arrays that the starter gets wrong.

## Performance — N = 50,000,000

| Implementation | Time | vs starter |
|---|---:|---:|
| starter (4-bit, 8 passes; wrong result for values ≥ 2^28) | 138.5 ms | 1.0× |
| **CUDA C++ final** | **49.1 ms** | **2.8×** |
| Numba CUDA | 65.3 ms | 2.1× |

Traffic is 4 passes × (200 MB histogram read + 200 MB scatter read + 200 MB
write) ≈ 2.4 GB, i.e. 49 GB/s effective.  A pure copy of the same bytes runs
at ~80 GB/s on this GPU; the difference is the radix scatter — its stores
leave in digit runs of only `BLOCK_ELEMS/256 = 8` elements (32 B), so every
store instruction breaks into several transactions — plus the per-element
ranking work.

The Numba port is unusually close to C++ here (1.3×) because the kernels are
long-running and memory-bound, so Numba's per-launch overhead and its 32-bit
histograms are a small part of the total.

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test radix_sort.cu test_harness.cu
```

`ITEMS` can be overridden with `-D` (the default is 8).

Numba:

```bash
python3 radix_sort.py
```

The script runs the correctness cases and the performance case.  For best
performance, pass Numba device arrays; host arrays incur copies to and from
the device.
