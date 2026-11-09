# Segmented Prefix Sum — CUDA C++ and Numba Results

Challenge: given `N` 32-bit floats and an integer `flags` array where
`flags[i] = 1` starts a new segment, compute the **exclusive** prefix sum
inside each segment,

```
output[i] = sum of values[j] for j < i with the same segment as i
```

(0 at every segment start).  `flags[0] = 1` always; `1 ≤ N ≤ 100,000,000`;
values are in `[-100, 100]`; performance is measured at `N = 50,000,000`.

Files:

- `segmented_prefix_sum.cu` — CUDA C++ implementation of
  `solve(values, flags, output, N)` (renamed from `starter.cu`).
- `segmented_prefix_sum.py` — Numba CUDA port with the same `solve` signature.

## Implementation — CUDA C++

The starter ran **three kernels plus a recursive scan over block
aggregates**, and materialised the whole inclusive scan in a `double` array
of length `N` (400 MB at `N = 50 M`).  Its traffic was well over a gigabyte
and it measured **39.5 ms** (≈15 GB/s of useful traffic).

The final kernel makes **one pass**: values and flags are read once, the
float output is written once, and no `N`-sized intermediate exists.  It is a
decoupled-lookback scan (the CUB `DeviceScan` design) generalised to the
segmented operator

```
seg_op(a, b) = ( b.flag ? b.val : a.val + b.val ,  a.flag | b.flag )
```

A pair `(val, flag)` summarises a whole range: `val` is the inclusive sum at
the range's last element, `flag` says whether any segment starts inside it.
That makes `seg_op` associative, which is what allows an arbitrary tree
(warp scans, block scans, cross-block lookback) to replace the starter's
strictly sequential recursion.

Structure:

| Detail | Value |
|---|---|
| Block | 256 threads, 2 chunks of 256 × 16 elements = **8192 elements** |
| Per thread | 16 consecutive elements per chunk, loaded as `float4` + `int4` |
| Warp scan | 5 shuffle rounds over `(val, flag)` pairs |
| Block scan | warp scans → warp 0 scans the 8 warp aggregates; **all chunks share both barriers**, so a block costs 3 `__syncthreads()` in total |
| Cross-block | decoupled lookback, one spinning thread per block |
| Lookback protocol | status 1 publishes the block **aggregate**, status 2 the block's **inclusive prefix**; `__threadfence()` between the payload and the status word |
| Output | written directly as `float`; accumulation in `double` |
| Scratch | aggregate/prefix/status arrays cached across `solve()` calls |

Two bugs found while developing it are worth recording, because both were
silent:

1. The first version published the block's **exclusive** prefix but still
   stopped the lookback at the first status-2 entry — so block 1 restarted its
   segment from 0.  Status-2 entries must hold `prefix + own aggregate`
   (inclusive), so one lookup fully resolves a predecessor.
2. Block 0 wrote `status[0] = 2` *before* `pre_val[0]`, so another block could
   observe a published prefix with stale contents.  The status word must be
   the last store after the fence.

### Tuning

Median of 10 runs at N = 50 M, `p = 0.001` (block elements = THREADS × ITEMS ×
CHUNKS):

| ITEMS | CHUNKS | THREADS | elems/block | registers | median |
|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 256 | 2,048 | 48 | 11.50 ms |
| 8 | 1 | 256 | 2,048 | 48 | 10.03 ms |
| 8 | 2 | 256 | 4,096 | 88 | 9.29 ms |
| 8 | 4 | 256 | 8,192 | 155 | 8.65 ms |
| 16 | 1 | 256 | 4,096 | 80 | 9.34 ms |
| 12 | 2 | 256 | 6,144 | 122 | 8.97 ms |
| **16** | **2** | **256** | **8,192** | **148** | **8.06 ms** |
| 32 | 1 | 256 | 8,192 | 146 | 9.72 ms |
| 16 | 2 | 128 | 4,096 | 148 | 8.40 ms |

The first version of the block scan synchronised once per chunk (8 barriers
per block); folding all chunks into the same two barriers and raising the
elements per block is what took the kernel from 12.9 ms to 8.1 ms.

## Implementation — Numba

The same lookback algorithm, with two Numba-specific changes:

1. **No register arrays.**  A thread cannot hold its 16 elements across the
   scan in registers, so the kernel runs two phases over the same data: the
   first computes the aggregate that feeds the scan, the second re-reads the
   elements (the block's 32 KB is still in L1/L2) and writes the outputs.
   DRAM traffic is unchanged.
2. **Coalescing drives the layout.**  Giving each thread 16 *consecutive*
   elements makes every warp load a stride-16 gather.  Measured for the reads
   alone: **16.3 ms strided vs 5.9 ms coalesced** (400 MB).  The kernel
   therefore splits each block into `CHUNKS` chunks of `THREADS × IPT`
   elements, where the `IPT` elements of a thread are consecutive but the
   loads of a chunk are coalesced across threads.

Every chunk still gets its own warp scan and all chunks share the same two
barriers, so the block costs three `__syncthreads()` here as well.

Tuning (median, N = 50 M):

| IPT | CHUNKS | elems/block | p = 0.001 | p = 0.1 |
|---:|---:|---:|---:|---:|
| 1 | 16 | 4,096 | 40.0 ms | 41.1 ms |
| 2 | 16 | 8,192 | 26.1 ms | 27.3 ms |
| **4** | **8** | **8,192** | **18.6 ms** | **19.2 ms** |
| 4 | 16 | 16,384 | 18.9 ms | 19.6 ms |
| 4 | 32 | 32,768 | 19.9 ms | 19.8 ms |
| 8 | 8 | 16,384 | 19.9 ms | 21.2 ms |
| 8 | 16 | 32,768 | 21.2 ms | 20.8 ms |
| 16 | 16 | 65,536 | 25.3 ms | 26.1 ms |

`IPT = 1` is instruction-bound (the segmented warp scan costs 5 double-
precision rounds per element) and `IPT ≥ 8` is memory-bound again (wider
strided reads); `IPT = 4` sits between the two.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, 6 SMs) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Kernel times are medians of 10 runs after a warm-up, with inputs already on
the device, including `cudaDeviceSynchronize()`.

## Correctness — CUDA C++

Reference: the sequential float64 recurrence on the CPU.  Because the kernel
also accumulates in double and in the same order, every case is **bit-exact**:

| Case | N | max abs error |
|---|---:|---:|
| challenge example (`[1,2,3,4,5,6]`, flags `[1,0,0,1,0,1]` → `[0,1,3,0,4,0]`) | 6 | 0 |
| single element | 1 | 0 |
| few segments | 1,000 | 0 |
| many segments | 1,000 | 0 |
| long segments (block boundary at 8,192) | 4,097 | 0 |
| 100k | 100,000 | 0 |
| 1M | 1,048,576 | 0 |
| every element its own segment | 10,000 | 0 |
| one single segment | 1,000,000 | 0 |

At the benchmark size the output is checked at sampled positions against a
float64 CPU recomputation: relative error 3.9e-08 (`p = 0.001`) and 2.7e-08
(`p = 0.1`) — float32 output rounding, which is all that remains.

## Correctness — Numba CUDA

All eleven cases pass: relative errors from 2.6e-08 to 4.5e-08; the largest
absolute differences (up to 3.9e-03) appear only where the expected value is
in the thousands, i.e. they are float32 output rounding.

## Performance — N = 50,000,000

Traffic is 12 bytes per element: 4 B value + 4 B flag read, 4 B output
written — 600 MB in total.

| Implementation | p = 0.001 | p = 0.1 | effective bandwidth | speedup |
|---|---:|---:|---:|---:|
| starter | 39.5 ms | 39.5 ms | 15.2 GB/s | 1.0× |
| **CUDA C++ final** | **8.57 ms** | **8.57 ms** | **70.0 GB/s** | **4.6×** |
| Numba CUDA | 21.0 ms | 19.5 ms | 28.5–30.8 GB/s | 2.0× |

The time is independent of the segment density (the per-element work is the
same either way), and the median run-to-run spread is about 0.05 ms.

A pure streaming kernel with the identical access pattern (read `float4` +
`int4`, write `float4`, same block geometry, no scan) measures **7.16 ms
(83.7 GB/s)** on this GPU, and the same kernel with the scan's double-
precision recurrence per element also measures 7.16 ms — the arithmetic is
free; memory is the wall.  The final kernel is at **84% of that roofline**
(90% counting the best run, 7.93 ms).  The remaining gap is the block scan's
barriers and the cross-block lookback, i.e. the price of the scan itself.

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test segmented_prefix_sum.cu test_harness.cu
```

`THREADS`, `ITEMS` and `CHUNKS` can be overridden with `-D` (the defaults are
256 / 16 / 2).

Numba:

```bash
python3 segmented_prefix_sum.py
```

The script runs the correctness cases and the two performance cases.  For
best performance, pass Numba device arrays; host arrays incur copies to and
from the device.
