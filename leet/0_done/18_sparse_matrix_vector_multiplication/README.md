# Sparse Matrix-Vector Multiplication — CUDA C++ and Numba Results

Challenge: `y_i = Σ_j A_ij · x_j` for a sparse matrix `A` that is **stored
densely in row-major order** (60–70% of its entries are zero) and a dense
vector `x`.  `1 ≤ M, N ≤ 10,000`; performance is measured at `M = 1,000`,
`N = 10,000`.

Files:

- `sparse_matrix_vector_multiplication.cu` — CUDA C++
  implementation of `solve(A, x, y, M, N, nnz)` (renamed from `starter.cu`).
- `sparse_matrix_vector_multiplication_naive.cu` — the naive baseline: one
  thread per row.
- `sparse_matrix_vector_multiplication.py` — Numba CUDA port.

## Where the time goes at M = 1,000, N = 10,000

`A` is 1,000 × 10,000 floats = **40 MB**, and because it is stored densely
every one of those bytes has to be read no matter how many entries are zero.
This is a pure streaming problem: the only thing that matters is whether the
kernel can keep the memory system busy.  A pure-read kernel over the same
40 MB measures **0.464 ms (86.2 GB/s)**, which is the floor for this
challenge on this GPU.

The starter's mapping (one warp per row, `float4` loads, a per-component
`if (a != 0)` before each `x` lookup) is already coalesced and reaches
**80–82 GB/s**, so there is no large algorithmic win available here — the
remaining ~6% is instruction overhead and memory-level parallelism.

## Implementation — CUDA C++

One 256-thread block per row.  Each block walks its row with `float4` loads:
256 threads × 16 B = 4 KB of contiguous row per iteration (the row is 40 KB,
so 10 iterations).  `x` is loaded as `float4` too; it is only 40 KB, so those
reads are L1/L2 hits, and one vector load replaces four scalar ones.  The
block reduces with a warp shuffle tree plus a cross-warp reduction in shared
memory, and thread 0 writes `y[row]`.

The inner loop is **branchless**: a zero entry contributes `0 * x`, which is
exact, so the per-element zero checks only save L1 lookups — and the sweep
below shows they cost more than they save at every sparsity level.

Dispatch in `solve`:

- `N % 4 == 0` and `A` 16-byte aligned → the vectorized kernel (all rows are
  a whole number of `float4`s and row pointers stay 16-byte aligned).
- `x` 16-byte aligned → `x` is read as `float4`; otherwise the same kernel
  with four scalar `x` loads (a template parameter).
- otherwise (any `N`, any row alignment) → a scalar block-per-row kernel.

The benchmark shape takes the fastest path.  `nnz` is accepted for signature
parity but not needed: it cannot reduce the traffic through a dense `A`.

### What was swept (M = 1,000, N = 10,000, 65% zeros, medians)

| Variant | Time | GB/s |
|---|---:|---:|
| pure-read kernel (floor) | 0.464 ms | 86.2 |
| warp per row, float4 A, scalar x (starter shape) | 0.497 ms | 80.6 |
| block per row, 256 thr, float4 A, scalar x | 0.474 ms | 84.4 |
| block per row, 256 thr, float4 A + float4 x | **0.465 ms** | **86.1** |
| block per row, 256 thr × 2 rows, float4 A + x | 0.466 ms | 85.9 |
| block per row, 512 thr × 2 rows, float4 A + x | 0.466 ms | 85.9 |
| block per row, 512 thr, float4 A + x | 0.505 ms | 79.1 |

Branchy (`if (a != 0)`) vs branchless, `float4` block-per-row at 512×2:

| Zeros | branchy | branchless |
|---|---:|---:|
| 0% | 81.0 GB/s | 85.0 GB/s |
| 50% | 82.0 GB/s | 85.3 GB/s |
| 65% | 82.2 GB/s | 85.4 GB/s |
| 90% | 81.7 GB/s | 85.3 GB/s |
| 99% | 80.5 GB/s | 85.3 GB/s |
| 100% | 81.4 GB/s | 85.4 GB/s |

Branchless is faster at every sparsity level, which is why `nnz` does not
select between the two forms.

### Naive baseline

`sparse_matrix_vector_multiplication_naive.cu` is the plainest correct form:
one thread per row, each walking its whole row with scalar global loads and a
serial accumulation — no vector types, no shared memory, no warp intrinsics.
It is **3.3× slower** (1.546 ms, 25.9 GB/s) because only 1,000 threads run,
each streaming 40 KB with a 40 KB-strided (uncoalesced) access pattern, and
the serial float32 accumulation also makes it the least accurate variant
(rel 1.7e-06 vs 1.3e-07 for the optimized kernel).

## Implementation — Numba

The same one-block-per-row mapping with scalar loads (Numba has no `float4`
load type); consecutive threads read consecutive elements, so the accesses
are still fully coalesced.  The inner loop is branchless and the block
reduction uses `shfl_down_sync` plus shared memory.  It reaches **0.538 ms
(74.3 GB/s)** — 1.16× the C++ kernel.  Kernels use `cache=True`.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, 6 SMs) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Times are medians after a warm-up, including `cudaDeviceSynchronize()`.  The
three C++ variants were compiled against the same harness binary and run
sequentially on an idle GPU; run-to-run variation is about ±2% (the optimized
kernel measured 0.465–0.476 ms across repeated runs).

> The starter as shipped does not compile: it uses `uintptr_t` without
> including `<cstdint>`.  Its numbers here were taken with that include
> added and nothing else changed.

## Correctness — CUDA C++

Against a float64 CPU reference (`rel = max|err| / max(1, max|ref|)`):

| Case | Result |
|---|---:|
| challenge example (3×4 → `[9, 13, 16]`) | exact |
| 1×1 / all-zero 2×2 / all-ones 2×2 / 3×2 | exact |
| 1×1000 (65% zeros) | rel 1.2e-07 |
| 999×1000 (65% zeros) | rel 9.8e-08 |
| 37×1001 / 37×1002 / 37×1003 (odd N, scalar path) | rel 1.3e-07 / 9.4e-08 / 1.1e-07 |
| 999×9999 (odd N, scalar path) | rel 1.1e-07 |
| 37×1000, `x+1` (misaligned x) | rel 8.9e-08 |
| 37×1000, `A+1` (misaligned A) | rel 1.8e-07 |
| 37×1001, `A+2`, `x+1` (both misaligned) | rel 1.0e-07 |
| benchmark shape | rel 1.3e-07 |

The non-zero errors are float32 accumulation-order differences; all cases are
far inside the 1e-5 tolerance used by the harness.

## Correctness — Numba CUDA

All twelve cases pass against a float64 NumPy reference (rel ≤ 1.4e-07),
including the challenge example, odd `N` (1001/1002/1003), 1×1000, 999×1000
and the benchmark shape.

## Performance — M = 1,000, N = 10,000, 65% zeros

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| `..._naive.cu` (one thread per row, serial) | 1.546 ms | 25.9 | 3.32× slower |
| starter (warp per row, `float4`, zero checks) | 0.492 ms | 81.3 | 1.06× slower |
| **`sparse_matrix_vector_multiplication.cu`** | **0.466 ms** | **85.9** | 1.0× |
| Numba CUDA | 0.538 ms | 74.3 | 1.16× slower |
| *pure-read floor* | *0.464 ms* | *86.2* | *1.00×* |

The optimized kernel is at the memory read ceiling: there is nothing else to
win from the 40 MB of traffic, and the 1.06× over the starter is the whole
available gain for this shape.

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test sparse_matrix_vector_multiplication.cu test_harness.cu
```

Numba:

```bash
python3 sparse_matrix_vector_multiplication.py
```

The script runs the correctness cases and the performance case.  For best
performance, pass Numba device arrays; host arrays incur copies to and from
the device.
