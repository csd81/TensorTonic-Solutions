# Sorting — CUDA C++ and Numba Results

Challenge: sort an array of 32-bit floats in ascending order, in place.  Any
algorithm is allowed.  `1 ≤ N ≤ 1,000,000`; performance is measured at
`N = 1,000,000`.

Files:

- `sorting.cu` — CUDA C++ implementation of `solve(data, N)` (renamed from
  `starter.cu`).
- `sorting.py` — Numba CUDA port with the same signature.

## The starter

A bitonic sort with **one kernel launch per compare-exchange step**.  At
N = 1,000,000 the padded power-of-two grid needs `log₂M · (log₂M + 1) / 2 =
210` steps, and each step is a full read-modify-write pass over the 8 MB
array — 3.4 GB of traffic through data that is re-read 210 times.  It
measures **17.0 ms**.

## Implementation — CUDA C++

The final version radix-sorts the IEEE-754 bit patterns: four 8-bit passes
instead of 210.

```
key(f) = sign(f) ? ~bits(f) : bits(f) | 0x80000000
```

Positive floats sort by their raw bits (which are monotonic in the value),
negative floats sort by the complement, and the inverted sign bit makes every
negative key smaller than every positive one.  `-0.0` sorts before `+0.0`,
which is a valid ascending order for values that compare equal.

The key is sorted as an unsigned integer with the same kernels as the integer
radix-sort challenge: per-block digit histograms, a digit-major scan of the
`(256 × nblocks)` matrix, and a stable scatter that ranks inside the block
with eight ballots per warp item and stages the writes through shared memory
so the global stores leave in digit-run order.  The **last pass converts the
key back to a float as it stores**, so there is no separate final copy.
Traffic is 8 MB for the transform plus 4 × 12 MB for the passes — about
56 MB against the starter's 3.4 GB.

Per-stage timing at N = 1 M (one pass): histogram 0.078 ms, scan 0.018 ms,
scatter 0.223 ms; the transform is ~0.1 ms.  The scatter dominates because
its stores leave in digit runs of `BLOCK_ELEMS / 256 = 8` elements.

### Tuning

| ITEMS per thread | 4 | **8** | 16 | 32 |
|---|---:|---:|---:|---:|
| time at N = 1 M | 1.49 ms | **1.35 ms** | 1.66 ms | 2.03 ms |

Larger tiles lengthen the digit runs (better stores) but cost occupancy and
registers; 8 wins at this size.

## Implementation — Numba

The same radix pipeline and the same key trick, with the bit-cast done by
`array.view(np.uint32)` inside the kernels, which Numba supports.  It reaches
1.82 ms — unusually close to C++ because the radix kernels are long-running
and the 17 launches are a small part of the total.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, 6 SMs) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Times are medians of 10–20 runs after a warm-up, including
`cudaDeviceSynchronize()`.

## Correctness — CUDA C++

Every case is compared element-by-element against `std::sort`; all match
exactly:

| Case | N |
|---|---:|
| challenge example (`[5,2,8,1,9,4]` → `[1,2,4,5,8,9]`) | 6 |
| single element | 1 |
| two elements, reversed | 2 |
| already sorted / reversed | 4 |
| duplicates | 5 |
| signed zeros (`+0.0`, `-0.0`) | 4 |
| negatives and mixed magnitudes (`-1e-8` to `1e8`) | 7 |
| random | 1,000, 100,000 |
| all positive | 1,000 |
| odd / power-of-two sizes | 1,023, 1,024 |
| benchmark size | 1,000,000 |

## Correctness — Numba CUDA

All thirteen cases match as well.

## Performance — N = 1,000,000

| Implementation | Time | vs starter |
|---|---:|---:|
| starter (bitonic, 210 launches) | 16.8–17.0 ms | 1.0× |
| **CUDA C++ final** (4-pass radix on float keys) | **1.23–1.35 ms** | **12.6–13.7×** |
| Numba CUDA | 1.82 ms | 9.4× |

(The C++ range is run-to-run variation across processes; an alternating
head-to-head measures 1.228/1.231 ms against the starter's 16.831/16.775 ms.)

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test sorting.cu test_harness.cu
```

`ITEMS` can be overridden with `-D` (the default is 8).

Numba:

```bash
python3 sorting.py
```

The script runs the correctness cases and the performance case.  For best
performance, pass Numba device arrays; host arrays incur copies to and from
the device.
