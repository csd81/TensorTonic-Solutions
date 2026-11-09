# Count Array Element — CUDA C++ and Numba Results

Challenge: count how many of the `N` `int32` values in `input` equal `K`, and
write the count to `output[0]`.  `1 ≤ N ≤ 100,000,000`; performance is
measured at `N = 100,000,000`, `K = 501,010`.

Files:

- `count_array_element.cu` — CUDA C++ implementation of
  `solve(input, output, N, K)` (renamed from `starter.cu`, an empty stub).
- `count_array_element.py` — Numba CUDA port with the same `solve` signature.

## Implementation — CUDA C++

This is a pure streaming reduction: the kernel must read 400 MB at the
benchmark size and nothing can avoid that, so the implementation is built for
load bandwidth:

- **`int4` (16-byte) loads** with a grid-stride loop — a warp reads 512
  contiguous bytes per instruction;
- each thread counts matches in registers, then a **warp shuffle reduction**
  and a per-warp shared-memory reduction collapse the block to one value;
- one `atomicAdd` per block into `output[0]` (4,096 blocks at most, so the
  atomic contention is negligible);
- `solve()` zeroes `output[0]` with `cudaMemsetAsync` before the kernel;
- the `N % 4` tail elements are handled by the first threads of block 0, and
  an unaligned input pointer falls back to a scalar-load kernel.

Launch configuration: 256 threads/block, grid capped at 4,096 blocks.

## Implementation — Numba

The same reduction (grid-stride loop, shuffle + shared block reduction, one
`cuda.atomic.add` per block) with scalar loads, since Numba cannot issue
`int4` vector loads.  A one-thread `zero_kernel` clears `output[0]` — writing
`output[0] = 0` from the host would cost a synchronous H2D scalar copy
(~0.27 ms on this machine).

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, sm_61) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Kernel times are medians of 20 runs (C++) / 10 runs (Numba) after a warm-up,
with the input already on the device, including `cudaDeviceSynchronize()`.

## Correctness — CUDA C++

`output[0]` is poisoned with `0xFF` before each run, so the kernel must write
the result.  Every case matches the CPU count exactly:

| Case | N | K | Count |
|---|---:|---:|---:|
| challenge example 1 | 5 | 1 | 2 |
| challenge example 2 | 4 | 11 | 0 |
| single match / miss | 1 | 7 / 8 | 1 / 0 |
| small all match | 4 | 4 | 4 |
| odd count | 5 | 4 | 5 |
| below one block | 1,023 | 12,345 | 1 |
| exact block | 1,024 | 999 | 2 |
| 100k sparse | 100,000 | 501,010 | 14,286 |
| 1M | 1,000,003 | 1 | 14 |
| 5M all match | 5,000,000 | 12,345 | 5,000,000 |
| benchmark | 100,000,000 | 501,010 | 0 |

The benchmark output was re-verified after timing (0, as expected).

Note: at the benchmark configuration `K = 501,010` lies outside the input
range `[1, 100,000]`, so the true count is zero.  The kernel does **not**
short-circuit on that — it still streams the whole array, which is the
workload being measured.

## Correctness — Numba CUDA

Same cases, all matching the CPU count (the 1M case uses a different random
draw, hence 10 instead of 14).

## Performance — N = 100,000,000 (400 MB of reads)

| Implementation | Median time | Read bandwidth |
|---|---:|---:|
| **CUDA C++ final** | **4.62 ms** | **86.6 GB/s** |
| Numba CUDA | 4.66 ms | 85.8 GB/s |

Both implementations are at the GPU's practical DRAM ceiling (the vector-add
and reduction challenges peak at ~84 GB/s; the `int4` loads reach a hair
above that).  There is no headroom left: the 400 MB input must be read once,
and both versions do exactly that.  The Numba port matches the hand-written
CUDA because even scalar loads saturate the bus for this access pattern.

## Build / run

CUDA C++ (the file exports `solve`, so compile it with a harness that
provides `main`):

```bash
nvcc -O3 -arch=sm_61 -o test count_array_element.cu test_harness.cu
```

Numba:

```bash
python3 count_array_element.py
```

The Python script runs the correctness cases and the performance case.  For
best performance, pass Numba device arrays; host arrays incur copies to and
from the device.

## Naive baseline

`count_array_element_naive.cu` is the same computation written as plainly as possible:
one thread per input element with an atomicAdd on a match — no shared-memory tiling, no vector types, no warp intrinsics and no
fast-math.  It passes the same correctness cases as the optimized kernel
(checked against a float64 CPU reference in the shared harness).

| Implementation | Time | vs naive |
|---|---:|---:|
| `count_array_element_naive.cu` | 5.037 ms | 1.00x |
| `count_array_element.cu` (optimized) | 4.629 ms | 1.09x |

Both are read-bound at 400 MB; the optimized kernel merely reduces the atomic traffic with vector loads and one atomic per block.
