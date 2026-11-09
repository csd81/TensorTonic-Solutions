# INT4 Matmul (W4A16) — CUDA C++ and Numba Results

Challenge: `y = x · dequant(W)ᵀ` with `x [M,K]` fp16, packed INT4 weights
`w_q [N, K/2]` uint8 (high nibble = `w[n,2i]`, low = `w[n,2i+1]`, signed
value `nibble − 8`), group-wise fp16 scales
(`W[n,k] = (nibble − 8) · scales[n, k/group_size]`) and fp16 output `y [M,N]`.
`M, N, K ≤ 8192`; performance is measured at `4096³`, `group_size = 128`.

Files:

- `int4_matmul.cu` — CUDA C++ implementation of
  `solve(x, w_q, scales, y, M, N, K, group_size)` (renamed from `starter.cu`).
- `int4_matmul_naive.cu` — the naive baseline: one thread per output.
- `int4_matmul.py` — Numba CUDA port with the same signature.

## Where the time goes at 4096³

This is 137 GFLOP on ~40 MB of data — **compute-bound**.  The starter runs one
thread per output with a K-loop that re-reads `x`, the packed weights and the
scales from global memory for every output element, and measures
**6.1 seconds (0.023 TFLOP/s)**.

| Variant | Time | TFLOP/s |
|---|---:|---:|
| starter (one thread per output) | 6105.9 ms | 0.023 |
| naive (same shape) | 6096.8 ms | 0.023 |
| **shipped (128×128 tiles, 8×8 register block)** | **160.2 ms** | **0.858** |
| Numba CUDA (32×32 tiles, dequant in shared) | 1049.0 ms | 0.131 |

**38×** faster.  0.858 TFLOP/s is the same class as the fp32 GEMM challenge
(0.955 TFLOP/s at 1024³) — the packing does not change the arithmetic, it
just makes the weight traffic a quarter of the fp32 case.

## Implementation — CUDA C++

The register-blocked GEMM from the GEMM/LoRA challenges, adapted to the packed
format:

- 128×128 block tile, 16×16 threads, 8×8 per thread (64 FP32 accumulators),
  interleaved thread columns (conflict-free shared reads).
- The `x` tile is loaded as fp16 and converted to float in shared memory.
- The weight tile is unpacked from the bytes, scaled and stored as float in
  shared memory: each byte serves two consecutive `k`, and the scale index is
  `n·(K/group_size) + k/group_size`.
- The epilogue converts the accumulators to fp16.

### Naive baseline

`int4_matmul_naive.cu` is the plainest correct form — the starter's shape:
one thread per output, a K-loop over packed bytes with global loads of `x`,
weights and scales.  It measures **6096.8 ms**, 38× slower.

## Implementation — Numba

The 32×32 shared-tile GEMM with one output per thread, dequantizing the weight
tile into shared memory exactly like the C++ loader (Numba cannot hold the
8×8 register tile).  Numba cannot index float16 arrays, so `x`, `scales` and
`y` are passed as int16 views and converted with `libdevice.half2float` /
`float2half_rn`.  It measures **1049 ms (0.131 TFLOP/s)**.  Kernels use
`cache=True`.

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

Checked against a float64 CPU reference built from the *host* dequantization
of the same packed bytes and scales:

| Case | Result |
|---|---|
| challenge example (2×4×4, group 2) | exact |
| 1×1×2, 3×5×4, 16×16×16, 33×17×32, 64×64×64, 128×32×128 at group sizes 2, 8, 32, 128 | max_abs ≤ 2e-03 |
| benchmark 4096³ — 200 randomly sampled (row, col) pairs in float64 | max_abs ≤ 1.4e-02 |

The errors are fp16 output rounding; the full 4096³ reference (137 GFLOP in
double) is impractical on the CPU, so that case is sample-checked.

## Correctness — Numba CUDA

All 23 small cases pass; the benchmark's 50 sampled pairs match.

## Performance — 4096 × 4096 × 4096, group 128

| Implementation | Time | TFLOP/s | vs optimized |
|---|---:|---:|---:|
| starter (untiled) | 6105.9 ms | 0.023 | 38.1× slower |
| `int4_matmul_naive.cu` (same shape) | 6096.8 ms | 0.023 | 38.1× slower |
| **`int4_matmul.cu`** (tiled, 8×8 register block) | **160.2 ms** | **0.858** | 1.0× |
| Numba CUDA (32×32 tile) | 1049.0 ms | 0.131 | 6.5× slower |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test int4_matmul.cu test_harness.cu
```

Numba:

```bash
python3 int4_matmul.py
```

The script runs the correctness cases and the performance case.
