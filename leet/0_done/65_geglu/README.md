# GEGLU — CUDA C++ and Numba Results

Challenge: GEGLU on a float32 vector — split `x` into halves
`x1 = x[0..N/2)`, `x2 = x[N/2..N)` and compute
`out[i] = x1[i] * GELU(x2[i])` with
`GELU(y) = 0.5·y·(1 + erf(y/√2))`.  `1 ≤ N ≤ 1,000,000`, `N` even;
performance is measured at `N = 1,000,000`.

Files:

- `geglu.cu` — CUDA C++ implementation of `solve(input, output, N)` (renamed
  from `starter.cu`).
- `geglu_naive.cu` — the naive baseline: one thread per output, scalar
  loads, `erff`.
- `geglu.py` — Numba CUDA port with the same signature.

## Where the time goes at N = 1,000,000

The kernel reads 4 MB (both halves) and writes 2 MB — 6 MB — so it is
**memory-bound**:

| Baseline | Time | GB/s |
|---|---:|---:|
| empty kernel launch | 6.7 µs | — |
| same-traffic kernel (`out = x1 * x2`, float4) | 0.079 ms | 76 |
| starter (scalar, `erff`) | 0.0787 ms | 76.3 |
| naive (scalar, `erff`) | 0.0788 ms | 76.2 |
| **shipped (float4, `erff`)** | **0.0788 ms** | **76.2** |

Unlike SiLU/SWiGLU at 400–600 KB, this kernel is large enough that the
`erff` is completely hidden behind the memory traffic, and the scalar starter
is already at the copy floor.  Vectorization changes nothing measurable.  So
there is no performance claim here beyond "at the wall"; the shipped file
adds the general-case shape (float4 only when `halfN % 4 == 0` and the
pointers are aligned, a scalar kernel otherwise).

## Implementation — CUDA C++

One `float4` per thread with 256-thread blocks: four outputs per thread, two
vector loads, one vector store, `erff` per element.  Scalar fallback for
`halfN % 4 != 0` or misaligned pointers.

### Naive baseline

`geglu_naive.cu` is the plainest correct form: one thread per output, two
scalar loads, `erff` and plain arithmetic.  It measures **0.0788 ms** — the
same as the shipped kernel to three digits.  As with ReLU, leaky ReLU and the
matrix copy, the streaming mapping is already optimal; the naive form has no
inefficiency to expose when every access is coalesced and DRAM is the limit.

## Implementation — Numba

Scalar one-element-per-thread kernel.  It measures **0.1108 ms (54.1 GB/s)**,
1.4× behind the C++ kernel — surprisingly close for Numba, because at 6 MB
the Python launch overhead is a small fraction of the runtime.

One Numba-specific trap was worth documenting: `math.erf` computes in
**float64** in Numba (`inspect_types` shows the call returning a 64-bit
value), and this GPU's FP64 throughput is 1/32 of FP32, so the first port
measured **0.818 ms — 10× slower than the C++ kernel**.  Switching to
`numba.cuda.libdevice.erff` (the float32 erf) with float32 constants brought
it to 0.111 ms.  Kernels use `cache=True`.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, 6 SMs) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Times are medians after a warm-up, including `cudaDeviceSynchronize()`.

## Correctness — CUDA C++

Checked against a float64 CPU reference (`std::erf`):

| Case | Result |
|---|---|
| challenge example 1 (`[1,1]` → `[0.8413447]`) | max_abs 4.9e-04 |
| challenge example 2 (`[2,-1,1,0.5]` → `[1.6826895, -0.3457312]`) | max_abs 4.9e-04 |
| all zeros / all equal / extremes (±100) | max_abs ≤ 4.9e-04 |
| even sizes 2, 4, 6, 10, 1024, 1026, 4098, 1,000,000 | rel ≤ 5.0e-08 |
| misaligned `input+1`, `output+1` (N = 1000) | rel ≤ 5.0e-08 |
| misaligned `input+2`, `output+3` (N = 1002) | rel ≤ 5.0e-08 |
| benchmark shape, N = 1,000,000 | max_abs 4.9e-04 |

The absolute error is float32 `erff` vs float64 `erf` on outputs of
magnitude up to ~10⁴, i.e. ~5e-08 relative; all 16 cases pass.

## Correctness — Numba CUDA

All 12 cases pass, matching the C++ results to within float rounding
(max_abs ≤ 4.9e-04, rel ≤ 5.2e-08).

## Performance — N = 1,000,000

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| `geglu_naive.cu` (scalar) | 0.0788 ms | 76.2 | 1.00× |
| starter (scalar) | 0.0787 ms | 76.3 | 1.00× |
| **`geglu.cu`** (float4) | **0.0788 ms** | **76.2** | 1.0× |
| Numba CUDA (after `erff` fix) | 0.1108 ms | 54.1 | 1.41× slower |
| *same-traffic floor* | *0.079 ms* | *76* | *1.00×* |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test geglu.cu test_harness.cu
```

`THREADS` can be overridden with `-D` (the default is 256).

Numba:

```bash
python3 geglu.py
```

The script runs the correctness cases and the performance case.
