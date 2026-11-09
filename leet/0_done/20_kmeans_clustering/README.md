# K-Means Clustering — CUDA C++ and Numba Results

Challenge: k-means over 2D points.  Given `data_x`, `data_y`, initial
centroids and a maximum iteration count, repeatedly assign each point to its
nearest centroid and re-mean the centroids; write the final labels and the
final centroids.  `1 ≤ sample_size ≤ 1,000,000`, `1 ≤ k ≤ 1,000`; performance
is measured at `sample_size = 10,000`, `k = 5`, `max_iterations = 30`.

Files:

- `kmeans_clustering.cu` — CUDA C++ implementation of `solve(...)` (renamed
  from `starter.cu`).
- `kmeans_clustering.py` — Numba CUDA port with the same signature.

## Where the time goes at the benchmark size

10,000 points × 5 clusters × 30 iterations is about 1.5 M distance
evaluations — microseconds of arithmetic.  The starter measured **1,360 µs**
because every iteration is host-driven: an assign kernel, four
`cudaMemset`s, an accumulate kernel (one global `atomicAdd` per point per
field), an update kernel, and then a **blocking 4-byte copy to the host** to
test the convergence flag.  That is ~150 launches, 120 memsets and 30 full
device stalls per solve.

## Implementation — CUDA C++

The final version runs the whole clustering in **one cooperative launch**:

```
for each iteration:
  assign every point + accumulate the cluster sums
  grid.sync()
  re-mean the centroids, note whether anything moved
  grid.sync()
  stop when nothing moved
final assignment pass so the labels match the final centroids
```

Three things mattered for the time:

1. **Accumulation.**  A per-element `atomicAdd` into the cluster sums is the
   trap here.  With five clusters a warp's 32 atomics collapse onto five
   addresses, and in the degenerate case (every point landing in one cluster)
   10,000 atomics chase a single address — measured at ~500 µs *per
   iteration*.  For `k ≤ 8` each thread therefore keeps eight
   `(sum_x, sum_y, count)` triples in registers, updated with predicated adds,
   and the block reduces them with warp shuffles; only `blocks × k` values
   reach global memory.  Larger `k` spreads atomics over enough addresses and
   takes the shared-atomic path.
2. **Centroids in registers.**  Reading `cx[c]`/`cy[c]` from global memory
   inside the per-point loop cost ~16 µs per iteration; staging them in
   registers once per iteration removed that.
3. **Grid size.**  One block keeps the sync cheap but uses one SM out of six.
   Four blocks of 1024 threads is the measured optimum at this size; the host
   picks the grid from `sample_size × k` (one block for small problems, up to
   the co-residency limit for large ones).  A single-block grid also takes a
   plain `__syncthreads()` instead of the cooperative-groups barrier.

### Tuning path

Median of 50 runs at the benchmark size:

| Version | Time |
|---|---:|
| per-element shared atomics, 256 threads, 2 blocks | 2,556 µs |
| + register accumulators for k ≤ 8 | 632 µs |
| + centroids staged in registers | 379 µs |
| + single-block `__syncthreads` specialization, 512 threads | 287 µs |
| `__launch_bounds__` + 1024 threads | 278 µs |
| 4 blocks of 1024 (grid sync amortized) | 223 µs |
| + reduce only the k live bins | **200.6 µs** |

Phase profile at one block (cycles over 30 iterations, thread 0): assign
213 K, warp/block reduce 112 K, both syncs 8 K, update 30 K.  The syncs were
never the problem — the per-point work and the accumulation were.

## Implementation — Numba

Numba exposes no grid sync and no cooperative launch, so the port takes the
other route to avoiding per-iteration launches: a **single block** that runs
the whole iteration loop in-kernel with `__syncthreads()`.  At the benchmark
size that is ten points per thread per iteration.  The accumulation follows
the C++ kernel (local arrays for `k ≤ 8`, warp-shuffle reduction, shared
atomics for larger `k`).

A 1024-thread block exceeds the launch resource limit for the Numba-compiled
kernel (`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`); 512 threads works.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, 6 SMs) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Times are medians of 50 runs after a warm-up, including
`cudaDeviceSynchronize()`.

## Correctness — CUDA C++

Compared against a float64 CPU implementation of the same algorithm
(assign → mean → early stop, then a final assignment):

| Case | Result |
|---|---|
| challenge example (`(1,1) (2,2) (8,8) (9,9)`, centroids `(1,1) (8,8)`) → centroids `(1.5,1.5) (8.5,8.5)`, labels `0 0 1 1` | exact |
| n=1, k=1 | exact |
| k=1 | exact |
| 100 points, k=3 | exact |
| 1,000 points, k=5 | exact |
| 5,000 points, k=8 | exact |
| benchmark shape | labels identical, centroids to 1.2e-07 |
| 10,000 points, k=50 and k=1000 | labels identical |
| duplicates, k = n | exact |
| wide coordinate range | exact |
| 100,000 points, k=16 | 1 label of 100,000 differs (centroids to 4.2e-05) |

The single 100k-point label difference is float32 accumulation versus the
float64 reference; the starter, summing every point into one global float
with atomics, differs on 27 labels in the same test.

## Correctness — Numba CUDA

All twelve cases pass; the 100k-point case matches the reference's labels
exactly (the block-level reduction in shared memory happens to round the same
way in double-precision comparisons).

## Performance — sample_size = 10,000, k = 5, 30 iterations

| Implementation | Time | vs starter |
|---|---:|---:|
| starter (3 kernels + 4 memsets + host sync per iteration) | 1,360 µs | 1.0× |
| **CUDA C++ final** (one cooperative launch) | **200.6 µs** | **6.8×** |
| Numba CUDA (one single-block launch) | 974 µs | 1.4× |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test kmeans_clustering.cu test_harness.cu
```

`KM_THREADS` and `KM_BLOCK_TARGET` can be overridden with `-D` (the defaults
are 1024 threads and 16 distance evaluations per thread per iteration, which
selects four blocks at the benchmark size).

Numba:

```bash
python3 kmeans_clustering.py
```

The script runs the correctness cases and the performance case.  For best
performance, pass Numba device arrays; host arrays incur copies to and from
the device.
