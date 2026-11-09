# Batch Normalization — CUDA C++ and Numba Results

Challenge: batch-norm forward for `[N, C]` row-major tensors,
`y = gamma·(x−mu)/sqrt(var+eps) + beta` with per-channel statistics.
`N ≤ 10,000`, `C ≤ 1,024`, `eps = 1e-5`; benchmark **N = 5,000**.

Files:

- `batch_normalization.cu` — CUDA C++ implementation (renamed from
  `starter.cu`).
- `batch_normalization_naive.cu` — the naive baseline: the starter's kernel
  verbatim (one block per channel).
- `batch_normalization.py` — Numba CUDA port.

## Results at N = 5,000

Same-process interleaved medians:

| C | naive = starter | shipped | speedup | Numba port |
|---:|---:|---:|---:|---:|
| 128 | 0.562 ms | **0.114 ms** | 5.0× | 0.348 ms |
| 512 | 4.024 ms | **0.393 ms** | 10.2× | 0.689 ms |
| 1024 | 9.158 ms | **0.757 ms** | 12.1× | 1.239 ms |

At C=1024 the shipped kernel moves 61 MB (two reads + one write) at
80.8 GB/s — the memory wall; the starter reaches only 6.6 GB/s.

## Why the starter is slow

One block per **channel**: every lane in a block reads a different row, a
stride of C floats, so each 4-byte load is its own memory transaction (32
per warp request instead of one 128-byte line).  It also reads the input
three times (mean pass, variance pass, normalize pass).

## Implementation — CUDA C++

1. **Stats kernel**: blocks cover 64-row tiles; consecutive threads own
   consecutive channels, so every row read is coalesced.  Per-thread partials
   are accumulated and added into per-channel **double** sums with
   `atomicAdd` (~160 K atomics at the benchmark — FP64 is free at that
   count).
2. **Finalize kernel** (one block over the channels): `mu = sum/N`,
   `inv_std = rsqrt(sumsq/N − mu² + eps)`, and `scale[j] = gamma[j]·inv_std`
   is stored alongside `mu[j]`.
3. **Write pass**: `float4` loads/stores over the flattened tensor with
   `y = scale[j]·(x − mu[j]) + beta[j]`; the channel index of a 4-element
   chunk is a multiple of 4 whenever C % 4 == 0, and a scalar path covers
   other C.

Three precision traps were found and fixed while validating:

- **Folding the affine transform** into `x·scale + shift` (with
  `shift = beta − mu·scale`) cancels catastrophically when the output is
  small compared with `x·scale` — for N=1 the exact answer is `beta`, but
  the folded form computes it as the difference of two ~1e5 values (error
  ~1e-3).  The kernel therefore keeps `mu` separate and evaluates
  `scale·(x − mu) + beta`, which is exact when `x == mu`.
- The **one-pass variance** `sumsq/N − mu²` can round slightly negative when
  the spread is tiny (N=1); `rsqrtf` of a negative argument is NaN, so the
  variance is clamped at zero before adding `eps`.
- With tiny N (the harness's `3×1024` case) a single float partial carries a
  whole channel through the cancellation above and loses the variance:
  accumulation in double is selected when `N ≤ 256` (the only regime where
  it matters; it costs ~2 FP64 ops per element).  For N > 256 the per-thread
  float partial covers ≤ 64 terms and the atomics are double — measured
  error 3.3e-6 on outputs of magnitude ~20.

Correctness: both challenge examples, `1x1`, `1x7` (the case that exposed the
folded-form cancellation), `5x3`, `17x33`, `1000x64`, `10000x5` (max N),
`3x1024` (max C, double path), and the benchmark shape at C = 128/512/1024.

## Numba port

Same three-kernel structure with scalar loads (Numba has no float4), float64
per-channel accumulators and float64 variance arithmetic, `libdevice.rsqrtf`
and the subtract-form write pass.  Measured 1.239 ms at C=1024 (49.6 GB/s).
