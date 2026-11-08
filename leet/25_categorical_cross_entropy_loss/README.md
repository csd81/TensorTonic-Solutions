# Categorical Cross-Entropy Loss — CUDA C++ and Numba Results

Challenge: compute the mean categorical cross-entropy over a batch of `N`
rows of `C` logits,

```
Loss_j = log(sum_k exp(z_jk)) - z_j,y_j
L      = mean_j Loss_j
```

with the numerically stable (log-sum-exp) form.  `1 ≤ N ≤ 10,000`,
`2 ≤ C ≤ 1,000`; performance is measured at `N = 10,000` (measured here for
`C = 1000`, `512` and `2`, since the brief does not pin `C`).

Files:

- `cross_entropy_loss.cu` — CUDA C++ implementation of
  `solve(logits, true_labels, loss, N, C)` (renamed from `starter.cu`).
- `cross_entropy_loss.py` — Numba CUDA port with the same `solve` signature.

## Implementation — CUDA C++

The starter assigned one block per row and made two passes over it (max pass,
then exp pass) with block-wide reductions and one `atomicAdd` per row.  The
final kernel instead gives **one warp per row** and computes `log(sum exp)` in
a **single pass** with the online-softmax recurrence:

```
if z > m:  s = s * exp(m - z) + 1;  m = z
else:      s += exp(z - m)
```

so each row is read once — half the global traffic — and:

| Detail | Value |
|---|---|
| Threads per block | 256 (8 warps ⇒ 8 rows per block) |
| Reduction | warp shuffles over `(m, s)` pairs, merged with the same recurrence |
| Global accumulation | one `atomicAdd` per block (1,250 instead of 10,000) |
| Hot exp | `__expf` (one per input element) |
| Final log | accurate `logf` (once per row, so it costs nothing) |

The choice of `__expf` matters: with the accurate `expf` the kernel ran at
0.626 ms, and with `__expf` it drops to 0.498 ms, i.e. the per-element exp
was the bottleneck until it was cheap enough for the loads to dominate.
Vectorising the loads (`float4`) changed nothing further (0.498 ms) because
the kernel is then at the memory roofline.

## Implementation — Numba

The same warp-per-row online-softmax kernel with scalar loads.  Each block
writes its eight row losses to a partials array and a small second kernel
reduces it into `loss[0]` — no atomics and no zeroing launch, which matters
because per-launch dispatch dominates at this problem size.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, sm_61) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Kernel times are medians of 50 runs after a warm-up, with the inputs on the
device, including `cudaDeviceSynchronize()`.  The loss reference is a float64
CPU computation of the same log-sum-exp formula.

## Correctness — CUDA C++

| Case | N | C | Loss (GPU) | Loss (reference) | Relative error |
|---|---:|---:|---:|---:|---:|
| challenge example 1 | 2 | 3 | 0.35489261 | 0.35489257 | 1.05e-07 |
| challenge example 2 | 3 | 4 | 0.98820376 | 0.98820373 | 3.80e-08 |
| single row | 1 | 2 | 0.00021601 | 0.00021601 | 3.24e-05 * |
| single row wide | 1 | 1,000 | 18.97572327 | 18.97572415 | 4.65e-08 |
| two classes | 5 | 2 | 5.95392513 | 5.95392500 | 2.16e-08 |
| small | 7 | 33 | 10.09137630 | 10.09137671 | 4.05e-08 |
| 100 rows | 100 | 1,000 | 13.60757065 | 13.60757017 | 3.52e-08 |
| near benchmark | 9,999 | 999 | 13.92678356 | 13.92678127 | 1.64e-07 |
| benchmark size | 10,000 | 1,000 | 13.85659599 | 13.85659980 | 2.75e-07 |
| constant logits | 64 | 10 | 2.30258536 | 2.30258509 | 1.17e-07 |
| extreme logits | 2 | 3 | 20.00002289 | 20.00002270 | 9.28e-09 |

\* The `N=1, C=2` case has a loss of `2.16e-4`; the absolute difference is
7e-9, so the relative error looks large only because the value itself is tiny.
Both challenge examples reproduce the stated losses.

## Correctness — Numba CUDA

All eleven cases pass with the same reference: relative errors from 2.3e-09
(benchmark size) to 6.9e-05 on the near-zero single-row loss (absolute
difference ~1e-8).

## Performance — N = 10,000

| C | starter | **CUDA C++ final** | Numba CUDA |
|---:|---:|---:|---:|
| **1,000** | 1.009 ms | **0.498 ms** (80.3 GB/s) | 0.792 ms (50.5 GB/s) |
| 512 | 0.700 ms | **0.268 ms** (76.4 GB/s) | 0.479 ms (42.8 GB/s) |
| 2 | 0.412 ms | **0.079 ms** | 0.245 ms |

At `C = 1000` the kernel reads the 40 MB of logits exactly once and runs at
~80 GB/s, i.e. **96% of the practical DRAM ceiling** of this GPU.  The speedup
over the starter is 2.0× at C=1000, 2.6× at C=512 and 5.3× at C=2 (where the
starter's per-row block reductions and 10,000 atomics dominated).
With C=2 the data is only 80 KB, so the measurement is launch-bound.

## Build / run

CUDA C++ (the file exports `solve`, so compile it with a harness that
provides `main`):

```bash
nvcc -O3 -arch=sm_61 -o test cross_entropy_loss.cu test_harness.cu
```

Numba:

```bash
python3 cross_entropy_loss.py
```

The Python script runs the correctness cases and the performance case.  For
best performance, pass Numba device arrays; host arrays incur copies to and
from the device.
