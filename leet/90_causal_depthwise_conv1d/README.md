# Causal Depthwise Conv1d — CUDA C++ and Numba Results

Challenge: compute

```
out[b, l, d] = bias[d] + sum_{k<K} weight[d, k] * x[b, l-k, d]
```

over a channels-last `(B, L, D)` tensor, with `x[b, l-k, d] = 0` for
`l-k < 0` (causal padding).  `1 ≤ B ≤ 16`, `1 ≤ L, D ≤ 8,192`, `1 ≤ K ≤ 8`;
performance is measured at `B = 8, L = 2,048, D = 4,096, K = 4`.

Files:

- `causal_depthwise_conv1d.cu` — CUDA C++ implementation of
  `solve(x, weight, bias, output, B, L, D, K)` (renamed from `starter.cu`).
- `causal_depthwise_conv1d.py` — Numba CUDA port with the same signature.

The operation is pure streaming: every input element is used, every output
element is written, and there is no cross-channel work.  At the benchmark
size that is 268 MB read + 268 MB written = **537 MB**, i.e. a floor of about
6.4 ms at this GPU's practical ~84 GB/s.

## Implementation — CUDA C++

The starter already had the right idea: one thread owns four channels
(`float4`), walks a tile of `TILE_L = 8` sequence positions, and keeps a
sliding window in registers so each input is loaded once.  It measured
7.3 ms — ~90% of the roofline — so the available headroom was small.  (It
also does not compile as shipped: `uintptr_t` is used without `<cstdint>`.)

The final kernel keeps the design and changes three things:

| Change | Why |
|---|---|
| **Templated on K** (dispatched over `K = 1..8`) instead of a `MAX_K = 8` register array with `if (k < K)` masking | halves the live window + weight registers at K=4 (32 registers total), removes the masked FMAs |
| **Circular window** `win[(s-j) mod K]` instead of shifting the window right every step | the shift cost `K-1` register moves per step (~100 per tile); the circular index is a compile-time constant in the unrolled step loop, so it costs nothing |
| **`TILE_L` 8 → 256** | halo re-read drops from `(8+3)/8 = 1.38x` to `(256+3)/256 = 1.01x` |

The kernel uses `float4` loads/stores when `D % 4 == 0` and the pointers are
16-byte aligned, with a scalar fallback otherwise.  Boundary handling is one
zero-filled prefill of the `K-1` slots before the tile, and a per-step
`l < L` guard that is uniform across the block.

### Tuning

Median of 50 runs, benchmark shape, K = 4 (THREADS = 256):

| TILE_L | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| median | 8.00 ms | 7.87 ms | 7.15 ms | 6.79 ms | 6.66 ms | **6.61 ms** | 6.60 ms |
| min | 7.34 | 7.31 | 7.04 | 6.66 | 6.60 | 6.56 | 6.57 |

(The 8- and 16-step variants land *behind* the starter, because the starter's
window shift is cheaper than the extra halo traffic at those tile sizes.)
`TILE_L = 256` and `512` are within noise of each other; 256 is the default
because it keeps more blocks in flight for smaller `L`.

## Implementation — Numba

Numba has neither vector loads nor register arrays, so the port uses one
thread per channel and `TILE_L = 16` consecutive sequence positions, with K
scalar loads per output.  Consecutive threads handle consecutive `d`, so all
loads and stores are coalesced, and walking `l` in order keeps the `K-1`
reused rows in the block's L1.

It cannot match the C++ kernel: **every input element is loaded K times**
instead of once, because there is no register window to hold it.  The cost
scales with K exactly as that predicts — 13.7 ms at K=3 versus 17.8 ms at
K=4, a ratio of 1.30 against the 4/3 = 1.33 ratio of load counts.  Tile
sweeps (16/32/64 steps) change the time by under 5%, confirming the
redundant loads, not locality, are the limit.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, 6 SMs) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Kernel times are medians of 50 runs after a warm-up, with inputs already on
the device, including `cudaDeviceSynchronize()`.

## Correctness — CUDA C++

Reference: the float64 CPU convolution.  All cases pass with maximum
relative error at most 1.2e-07:

| Case | Shape | max rel error |
|---|---|---:|
| challenge example (`[1..8]`, weights `[1,0,-1]`/`[1,1,1]`) | B=1 L=4 D=2 K=3 | 0 (exact) |
| single element, K=1 | B=1 L=1 D=4 K=1 | 0 (exact) |
| L=1 with K=3 | B=1 L=1 D=4 K=3 | 0 (exact) |
| D not divisible by 4 (scalar path) | B=1 L=5 D=3 K=2 | 3.7e-08 |
| K=8 (maximum) | B=1 L=7 D=4 K=8 | 1.2e-07 |
| small | B=2 L=33 D=8 K=4 | 4.9e-08 |
| D=12, K=5 | B=3 L=129 D=12 K=5 | 8.2e-08 |
| D=17 (odd) | B=1 L=1000 D=17 K=3 | 7.9e-08 |
| B=16 (maximum) | B=16 L=64 D=256 K=4 | 6.3e-08 |
| D=1000 | B=2 L=300 D=1000 K=3 | 7.2e-08 |
| benchmark shape | B=3 L=2048 D=4096 K=4 | 1.2e-07 |

The challenge example reproduces the stated output exactly
(`[1,2],[3,6],[4,12],[4,18]`), including the zero-padded first two
positions.

## Correctness — Numba CUDA

All eleven cases pass; relative errors 4.2e-08 to 9.4e-08 (the same float32
rounding as the C++ kernel).

## Performance — B=8, L=2048, D=4096

| Implementation | K=4 | K=3 | effective bandwidth (K=4) |
|---|---:|---:|---:|
| starter | 7.29 ms | 7.17 ms | 73.6 GB/s |
| **CUDA C++ final** | **6.61 ms** | **6.65 ms** | **81.2 GB/s** |
| Numba CUDA | 17.8 ms | 13.7 ms | 30.2 GB/s |

The final kernel moves 537 MB in 6.61 ms = **81 GB/s, about 97% of the
~83.7 GB/s practical DRAM ceiling** measured on this GPU; the `TILE_L = 256`
halo adds only ~1%.  For comparison, a plain `float4` copy of the same
537 MB measured 6.98 ms with the naive one-thread-per-`float4` geometry, so
the convolution is *faster than that copy* — the copy simply had less
per-thread work to hide latency with.

The speedup over the starter is 1.10×, and the main practical difference is
consistency: the starter's median wanders between 6.7 and 7.4 ms across
processes, while the final kernel stays at 6.55–6.72 ms.  The starter was
already close to the memory wall, so there is not much more to win — with
537 MB of unavoidable traffic there is no algorithmic trick left.

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test causal_depthwise_conv1d.cu test_harness.cu
```

`TILE_L` and `THREADS` can be overridden with `-D` (the defaults are 256 and
256).

Numba:

```bash
python3 causal_depthwise_conv1d.py
```

The script runs the correctness cases and the two performance cases.  For
best performance, pass Numba device arrays; host arrays incur copies to and
from the device.
