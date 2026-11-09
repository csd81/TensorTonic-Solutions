# LoRA Linear — CUDA C++ and Numba Results

Challenge: `output = x·Wᵀ + lora_scale·(x·Aᵀ)·Bᵀ` with `x [M,K]`, `W [N,K]`,
`A [R,K]`, `B [N,R]`, all float32 row-major.  Benchmark:
`M = 256, K = 4096, N = 4096, R = 64` — 8.9 GFLOP, 97% of it in the base
`x·Wᵀ` product.

Files:

- `lora_linear.cu` — CUDA C++ implementation of
  `solve(x, W, A, B, output, batch, d_in, d_out, rank, lora_scale)`
  (renamed from `starter.cu`).
- `lora_linear_naive.cu` — the naive baseline: three untiled products, one
  thread per output element.
- `lora_linear.py` — Numba CUDA port with the same signature.
- `cuBLAS.cu` — a cuBLAS reference implementation (see the note below; it
  cannot be built on this machine and the challenge forbids external
  libraries, so it is not the submitted solution).

## Where the time goes at 256 × 4096 × 4096, R = 64

`x·Wᵀ` is a 256×4096×4096 GEMM (8.6 GFLOP); `x·Aᵀ` (256×4096×64) and
`temp·Bᵀ` (256×64×4096) add 0.27 GFLOP.  The problem is compute-bound.

| Variant | Time | TFLOP/s |
|---|---:|---:|
| starter (untiled, one thread per output) | 282.0 ms | 0.031 |
| naive (same shape) | 283.4 ms | 0.031 |
| **shipped (128×128 tiles, 8×8 per thread)** | **9.28 ms** | **0.955** |
| Numba CUDA (32×32 shared tiles, 1 output/thread) | 61.1 ms | 0.145 |
| *FP32 pipeline peak on this GPU (no tensor cores)* | — | *~2.27* |

**30.5×** over the naive form.  The starter re-reads every `W` row for every
output row with strided, uncoalesced accesses; the shipped kernel tiles the
operands through shared memory and holds an 8×8 accumulator block per
thread, which is the same structure that won the standalone GEMM challenge.

## Implementation — CUDA C++

One templated kernel `C = scale·A·Bᵀ` (or `C += scale·A·Bᵀ`) serves all
three products, because `W`, `A` and `B` are all stored `[N, K]` row-major:

- 128×128 block tile, 16×16 threads, 8×8 per thread (64 FP32 accumulators).
- Interleaved thread columns (`tx + j·16`) keep the shared-memory B-fragment
  reads conflict-free.
- The B tile is loaded with `k` in the inner dimension (consecutive
  `K`-elements are contiguous in the `[N, K]` layout).
- Phase 1 writes `output = x·Wᵀ`; phase 2 computes the small
  `temp = x·Aᵀ` (M×R, 64 KB, cached workspace); phase 3 accumulates
  `output += lora_scale·temp·Bᵀ`.

### Naive baseline

`lora_linear_naive.cu` is the plainest correct form — the starter's shape:
three untiled products, one thread per output, a K-loop that reads both
operand rows directly from global memory.  It measures **283.4 ms**, 30.5×
slower.

### Note on `cuBLAS.cu`

The directory also contains a cuBLAS reference implementation.  Two caveats:

1. **It cannot be built here**: this machine has no cuBLAS at all (no
   `cublas_v2.h` and no `libcublas*` anywhere), so it could not be validated
   or timed.  The file's `cublasSgemm` calls were corrected for the
   column-major/row-major mapping (a row-major `[rows, cols]` buffer read
   with `lda = rows` is its transpose, so each product uses
   `CUBLAS_OP_T` on the first operand and `CUBLAS_OP_N` on `x`/`temp`, with
   leading dimensions equal to the row lengths): the previous version read
   `W` as `Wᵀ` and computed `x·W`.
2. The challenge says "Do not use external libraries beyond those
   provided", so cuBLAS is not a valid submission regardless.  It would
   likely be the fastest option (~2 TFLOP/s class) if both issues were
   removed.

## Implementation — Numba

All three products run through one 32×32 shared-tile GEMM `C = A·Bᵀ` with one
output per thread (Numba cannot hold the 8×8 register tile — 64 accumulators
would spill to local memory).  It measures **61.1 ms (0.145 TFLOP/s)** — the
compute-bound analogue of the GEMM challenge's Numba gap.  Kernels use
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

Checked against a float64 CPU reference with per-element tolerance
`1e-3 + 1e-4·|ref|`:

| Case | Result |
|---|---|
| challenge example (2×4, W 3×4, A 2×4, B 3×2, scale 0.5) | exact |
| 1×1×1 r1; 3×10×7 r2; 17×33×19 r5; 64×128×64 r16; 100×257×129 r64; 256×512×512 r256 | rel ≤ 6.6e-07 |
| benchmark shape (256 × 4096 × 4096, r64) | rel 1.3e-06 |

All 8 harness cases pass.

## Correctness — Numba CUDA

All 6 cases pass with the same tolerance.

## Performance — 256 × 4096 × 4096, rank 64

| Implementation | Time | TFLOP/s | vs optimized |
|---|---:|---:|---:|
| starter (untiled) | 282.0 ms | 0.031 | 30.4× slower |
| `lora_linear_naive.cu` (same shape) | 283.4 ms | 0.031 | 30.5× slower |
| **`lora_linear.cu`** (tiled, 8×8 register block) | **9.28 ms** | **0.955** | 1.0× |
| Numba CUDA (32×32 tile, 1 output/thread) | 61.1 ms | 0.145 | 6.6× slower |
| `cuBLAS.cu` | — | — | not buildable here |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test lora_linear.cu test_harness.cu
```

Numba:

```bash
python3 lora_linear.py
```

The script runs the correctness cases and the performance case.
