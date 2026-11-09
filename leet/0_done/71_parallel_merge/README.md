# Parallel Merge — CUDA C++ and Numba Results

Challenge: merge two sorted float32 arrays `A [M]` and `B [N]` into a sorted
`C [M+N]`.  `M, N ≤ 50,000,000`, `M+N ≤ 50,000,000`; performance is measured
at `M = N = 25,000,000`.

Files:

- `parallel_merge.cu` — CUDA C++ implementation of `solve(A, B, C, M, N)`
  (renamed from `starter.cu`).
- `parallel_merge_naive.cu` — the naive baseline: one binary search per
  output element.
- `parallel_merge.py` — Numba CUDA port with the same signature.

## Where the time goes at M = N = 25,000,000

The kernel reads 200 MB and writes 200 MB — 400 MB — so the floor is ~4.9 ms
at this GPU's ~82 GB/s.  The starter does a **binary search per output
element** (50 M searches, each a ~25-probe random walk through the 100 MB
inputs) and measures **45.8 ms (8.7 GB/s)**.

| Variant | Time | Effective GB/s |
|---|---:|---:|
| starter (search per element) | 45.80 ms | 8.7 |
| naive (same shape) | 45.64 ms | 8.8 |
| **shipped (merge-path tiles, precomputed partitions)** | **7.05 ms** | **56.7** |
| Numba CUDA (search per element) | 55.8 ms | 14.3 |

**6.5×** over the naive form.  The road there, all measured on the same
harness:

| Step | Time |
|---|---:|
| per-thread merge-path (search per thread, no tiling) | 34.5 ms (best ITEMS=16) |
| block tile with the binary search *inside* each block | 22.3 ms |
| + partitions precomputed in a separate parallel kernel | 8.37 ms |
| + merge into registers with `float4` stores (TILE=3072) | **6.47–7.05 ms** |

The biggest single win was moving the per-block binary search out of the
merge kernel: a 25-probe dependent chain in front of every tile serialised
the block prologue.  Computing all ~16 K partitions in one kernel of
independent searches hides that latency completely.

## Implementation — CUDA C++

1. `partition_kernel` — one thread per block boundary computes the
   merge-path split `(i, j)`, `i+j = diagonal`, by binary search over `A`
   and `B` (the standard condition `A[mid] <= B[d-1-mid]`).
2. `merge_tile_kernel` — each block owns a 3072-element output tile.  It
   loads its `A`-range and `B`-range into shared memory with coalesced global
   reads, then each of the 256 threads binary-searches (in shared memory) the
   split of its own 12-output sub-range, merges 12 values into registers, and
   writes them as three coalesced `float4` stores.

TILE sweep: 512 → 10.7, 1024 → 8.0, 2048 → 6.6, **3072 → 6.47**,
4096 → 8.7 ms (the largest tile loses occupancy to shared memory).

### Naive baseline

`parallel_merge_naive.cu` is the plainest correct form — the starter's shape:
one binary search per output element with a grid-stride loop.  It measures
**45.6 ms**, 6.5× slower.

## Implementation — Numba

The per-element merge-path kernel (the C++ naive form): the tile pipeline
needs a partition kernel plus shared-memory staging with per-thread searches,
which is not practical to express compactly in Numba.  It measures
**55.8 ms (14.3 GB/s)**.  Kernels use `cache=True`.

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

The merge is exact, so outputs are compared element-for-element with
`std::merge`:

| Case | Result |
|---|---|
| challenge example 1 (`[1,3,5,7]` + `[2,4,6,8]`) | exact |
| challenge example 2 (`[-1,1,3]` + `[2]`) | exact |
| single/single | exact |
| A < B, B < A (disjoint ranges) | exact |
| interleaved random, all-equal values (50 + 70) | exact |
| 1000/999 random | exact |
| skewed 1 / 100,000 | exact |
| benchmark shape (25 M + 25 M) | exact |

All 10 harness cases pass (11 checks including the benchmark-shape output
check).

## Correctness — Numba CUDA

All 8 cases pass, exact.

## Performance — M = N = 25,000,000

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| starter (search per element) | 45.80 ms | 8.7 | 6.5× slower |
| `parallel_merge_naive.cu` (same shape) | 45.64 ms | 8.8 | 6.5× slower |
| **`parallel_merge.cu`** (tile merge path) | **7.05 ms** | **56.7** | 1.0× |
| Numba CUDA | 55.8 ms | 14.3 | 7.9× slower |
| *400 MB read+write floor* | *~4.9 ms* | *~82* | *0.7×* |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test parallel_merge.cu test_harness.cu
```

Numba:

```bash
python3 parallel_merge.py
```

The script runs the correctness cases and the performance case.
