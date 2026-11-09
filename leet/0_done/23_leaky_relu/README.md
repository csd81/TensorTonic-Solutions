# Leaky ReLU — CUDA C++ and Numba Results

Challenge: leaky ReLU with `alpha = 0.01`:
`out[i] = x > 0 ? x : 0.01 * x`.  `1 ≤ N ≤ 100,000,000`; performance is
measured at `N = 50,000,000`.

Files:

- `leaky_relu.cu` — CUDA C++ implementation of `solve(input, output, N)`
  (renamed from `starter.cu`).
- `leaky_relu_naive.cu` — the naive baseline: one thread per element, scalar
  loads/stores and a ternary.
- `leaky_relu.py` — Numba CUDA port with the same signature.

## Where the time goes at N = 50,000,000

The kernel reads 200 MB and writes 200 MB — 400 MB of traffic with no reuse
and a single multiply per element.  A pure `float4` copy of the same 400 MB
measures **4.860 ms (82.3 GB/s)**, and the starter measures **4.861 ms
(82.3 GB/s)**: it is already at the wall.  There is no arithmetic or
bandwidth headroom to find here; the work below adds the general-case
mechanics and fixes the starter's alignment assumption.

One real defect in the starter: it casts `input`/`output` to `float4` without
checking alignment.  With 16-byte-aligned pointers (which is what `cudaMalloc`
returns) it is fine, but either pointer at a non-multiple-of-16 offset makes
it fault with `misaligned address`.  The implementation below checks the
pointers and falls back to a scalar kernel.

## Implementation — CUDA C++

One `float4` per thread with 256-thread blocks (12.5 M threads in ~48.8 K
blocks at the benchmark size), one kernel launch, and the scalar tail
(`N % 4`) handled
in-kernel by the thread just past the last `float4`.  If either pointer is
not 16-byte aligned, a scalar one-thread-per-element kernel runs instead.

The block-size sweep through the harness gives 4.86–4.97 ms for 128/256/512
threads — all inside the ~1% run-to-run noise of a kernel sitting on the DRAM
wall — so 256 is kept.  The same sweep in challenge 21 (ReLU) showed the same
flatness, as expected for a pure streaming kernel.

### Naive baseline

`leaky_relu_naive.cu` is the plainest correct form: one thread per element,
plain scalar load/store, one ternary.  It measures **4.884 ms (81.9 GB/s)** —
0.4% behind the float4 kernel, i.e. the same number.  At this size the
coalesced scalar traffic saturates DRAM just as well as vectorized traffic;
as with ReLU, the naive mapping is already the optimal one.

## Implementation — Numba

Scalar one-element-per-thread kernel (no `float4` in Numba).  It reaches
**4.976 ms (80.4 GB/s)**, about 2.3% behind the C++ kernel — the launch
overhead that dominates at small sizes is negligible next to 4.9 ms of DRAM
traffic.  Kernels use `cache=True`.

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
C++ variants were compiled against the same harness and run sequentially on
an idle GPU; run-to-run variation is about ±1% at this size.

## Correctness — CUDA C++

The operation is exact per element, so every case is checked for bit-exact
equality against a float32 CPU reference (`x > 0 ? x : 0.01f * x`):

| Case | Result |
|---|---|
| challenge example 1 (`[1,-2,3,-4]`) | exact |
| challenge example 2 (`[-1.5,0,2.5,-3]`) | exact |
| single positive / negative / zero | exact |
| all negative / all positive / all zero | exact |
| sizes 2, 3, 4, 5, 7, 1023, 1024, 1025, 4099, 100,000 | exact |
| misaligned `input+1`, `output+1` (N = 1000) | exact |
| misaligned `input+2`, `output+3` (N = 1001) | exact |
| benchmark shape, N = 50,000,000 | exact |

The starter fails both misaligned cases with `misaligned address`; the
implementation above passes them through its scalar fallback.  (A faulting
kernel poisons the CUDA context, so the harness runs these probes after all
timing measurements.)

## Correctness — Numba CUDA

All eighteen cases pass bit-exact against the NumPy reference, including the
challenge examples, odd sizes, and N = 50,000,000.

## Performance — N = 50,000,000

| Implementation | Time | GB/s | vs optimized |
|---|---:|---:|---:|
| `leaky_relu_naive.cu` (scalar, 1 thread/element) | 4.884 ms | 81.9 | 1.00× |
| starter (float4, no alignment check) | 4.861 ms | 82.3 | 1.00× |
| **`leaky_relu.cu`** (float4, alignment fallback) | **4.866 ms** | **82.2** | 1.0× |
| Numba CUDA | 4.976 ms | 80.4 | 1.02× slower |
| *float4 copy floor* | *4.860 ms* | *82.3* | *1.00×* |

Everything on the GPU side is the same number within noise: this is a
DRAM-wall challenge, and the starter's mapping was already optimal.

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test leaky_relu.cu test_harness.cu
```

`THREADS` can be overridden with `-D` (the default is 256).

Numba:

```bash
python3 leaky_relu.py
```

The script runs the correctness cases and the performance case.  For best
performance, pass Numba device arrays; host arrays incur copies to and from
the device.
