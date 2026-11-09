# RoPE Embedding — CUDA C++ and Numba Results

Challenge: rotary positional embedding with the half-split layout —
`out[j] = q[j]·cos[j] − q[j+D/2]·sin[j]` and
`out[j+D/2] = q[j+D/2]·cos[j+D/2] + q[j]·sin[j+D/2]` — where the challenge
guarantees `cos[j] == cos[j+D/2]` and `sin[j] == sin[j+D/2]`.
`Q`, `cos`, `sin` and `out` are `M × D`, `D` even, `M, D ≤ 10,000`;
performance is measured at `M = 1,048,576`, `D = 128`.

Files:

- `rope_embedding.cu` — CUDA C++ implementation of
  `solve(Q, cos, sin, output, M, D)` (renamed from `starter.cu`).
- `rope_embedding_naive.cu` — the naive baseline: one thread per element,
  scalar loads of all three inputs.
- `rope_embedding.py` — Numba CUDA port with the same signature.

## Where the time goes at M = 1,048,576, D = 128

The starter reads all of `Q`, `cos` and `sin` (1.61 GB) and writes 536 MB —
**2.15 GB** — and measures **27.8 ms (77 GB/s)**.

The half-split guarantee means `cos`/`sin` only have to be read once per
half-pair: a thread loads a float4 of `Q` from the first half, the mirrored
float4 from the second half, and one float4 each of `cos` and `sin`, then
writes **both** output groups:

| Variant | Time | Effective GB/s |
|---|---:|---:|
| starter (scalar, full cos/sin) | 27.76 ms | 77 (of 2.15 GB) |
| naive (same shape) | 27.09 ms | 79 (of 2.15 GB) |
| **shipped (float4, half cos/sin)** | **19.9–20.5 ms** | **81 (of 1.61 GB)** |
| Numba CUDA (full cos/sin) | 27.39 ms | 78 (of 2.15 GB) |

Traffic drops to `Q` 536 MB + `cos`/`sin` 268 MB each + `out` 536 MB =
**1.61 GB**, and the kernel runs at ~81 GB/s of that — the same memory wall,
with a third less data moved.

## Implementation — CUDA C++

Each thread handles one 4-column group of a row: `float4` loads of `q1`
(first half), `q2` (second half), `cos`/`sin` (first half, reused for the
second half thanks to the invariant), two `float4` stores.

The fast path requires `D % 8 == 0` (so the half offset is a whole number of
float4s) and 16-byte-aligned pointers; a scalar elementwise kernel covers
everything else (and does not rely on the invariant).  Row blocks are mapped
to `blockIdx.x` because `gridDim.y` caps at 65,535 — with 1 M rows the row
count per grid is 65,536.

### Naive baseline

`rope_embedding_naive.cu` is the plainest correct form — one thread per
element with an integer modulus for the column, scalar loads of `Q`, `cos`,
`sin` and one store.  It measures **27.1 ms**, 1.36× slower.

## Implementation — Numba

Scalar elementwise kernel reading all three inputs (no float4, and the
half-split reuse is tied to the vectorized mapping here).  It measures
**27.39 ms (78.4 GB/s of 2.15 GB)** — the same as the C++ naive form.
Kernels use `cache=True`.

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

Checked against a float64 reference with per-element tolerance
`1e-4 + 1e-4·|ref|`:

| Case | Result |
|---|---|
| challenge example (2×4: identity row and pure-rotation row) | exact |
| D = 2, D = 6 (scalar path) | max_abs ≤ 1e-07 |
| 1×8, 1×96, 5×16, 17×64, 100×128, 3×130, 64×4096 | max_abs ≤ 2e-07 |
| benchmark shape (1,048,576 × 128) | max_abs ≤ 2e-07 |

All 10 harness cases pass.

## Correctness — Numba CUDA

All 10 cases pass with the same tolerance.

## Performance — M = 1,048,576, D = 128

| Implementation | Time | vs optimized |
|---|---:|---:|
| starter (scalar, full cos/sin) | 27.76 ms | 1.39× slower |
| `rope_embedding_naive.cu` (same shape) | 27.09 ms | 1.36× slower |
| **`rope_embedding.cu`** (float4, half cos/sin) | **19.9–20.5 ms** | 1.0× |
| Numba CUDA | 27.39 ms | 1.37× slower |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test rope_embedding.cu test_harness.cu
```

Numba:

```bash
python3 rope_embedding.py
```

The script runs the correctness cases and the performance case.
