# cuda-basics → Numba ports

Every problem directory now holds its original CUDA source and a Numba port
side by side:

```
<problem>/
├── <problem>.cu     # original CUDA C
├── <problem>.md     # problem statement
└── <problem>.py     # Numba port
```

Each `.py` contains:

- the kernel(s) as `@cuda.jit` functions,
- `solve(...)`, mirroring the C `extern "C" void solve(...)` argument list,
- `test()`, a self-check against a NumPy reference.

Run everything from this directory (`study-plans/cuda-basics`):

```bash
python run_all.py                    # all 34 ports
python run_all.py softmax argmax     # selected ports
python softmax/softmax.py            # a single port
```

Requirements: an NVIDIA GPU and a Numba with CUDA support
(`cuda.is_available()` must be `True`). On Colab, select a GPU runtime;
normally Numba ships with it. If the CUDA target is missing:

```python
!pip install -q -U numba numba-cuda
```

Then restart the runtime. Numba compiles for the attached GPU automatically —
there is no `-arch`/`sm_75` flag to pass.

## Calling convention

`solve` takes **device arrays**, exactly like the C version takes raw
pointers. Kernels use flat row-major indexing, so pass flattened arrays for
multi-dimensional problems:

```python
import numpy as np
from numba import cuda
from softmax.softmax import solve

d_x = cuda.to_device(x.reshape(-1))        # float32
d_out = cuda.to_device(out.reshape(-1))
solve(d_x, d_out, N)
cuda.synchronize()
out = d_out.copy_to_host().reshape(shape)
```

Results the C version returns through a pointer (`float* result`,
`int* result`) are one-element device arrays here.

## CUDA C → Numba mapping used in these ports

| CUDA C | Numba |
|---|---|
| `__global__ void k(...)` | `@cuda.jit` def k(...) |
| `k<<<blocks, threads>>>(...)` | `k[blocks, threads](...)` |
| `k<<<b, t, smem>>>(...)` | `k[b, t, 0, smem](...)` |
| `__shared__ float s[32]` | `cuda.shared.array(32, float32)` |
| `extern __shared__ float s[]` | `cuda.shared.array(0, float32)` + `sharedmem` launch arg |
| `__syncthreads()` | `cuda.syncthreads()` |
| `__shfl_down_sync(m, v, o)` | `cuda.shfl_down_sync(m, v, o)` |
| `atomicAdd(&p[i], v)` | `cuda.atomic.add(p, i, v)` |
| `atomicMax/Min` via CAS | `cuda.atomic.max/min` (native for float32) |
| `erff`, `tanhf`, `expf` | `libdevice.erff`, `math.tanh`, `math.exp` |
| `cudaMalloc` / `cudaMemcpy` | `cuda.device_array`, `cuda.to_device`, `copy_to_host` |

## Intentional differences from the `.cu` versions

- **No `float4`/vector types.** Numba has no `float4`, so the vectorized
  kernels (`relu`, `gelu`, `sigmoid`, `swish`, `tanh`, `leaky-relu`,
  `dropout`, `rms-norm`) are scalar/grid-stride equivalents. Math and
  results are the same.
- **float32 only.** Numba does not support bfloat16 and has limited float16
  arithmetic; every port uses float32 (the original platform tests also use
  float32 here).
- **`max(a, b)` / `min(a, b)` are not usable inside Numba 0.68 CUDA kernels.**
  The ports use comparison/ternary or warp-shuffle reductions instead.
- **Reduction order differs.** Warp/block trees and atomics match the C
  algorithm, but the exact summation order is not required to match; the
  tests use f32-appropriate tolerances.
- **Integer division must be `//`.** Python/Numba `/` produces a float.
- Scratch buffers that the C code `cudaMalloc`/`cudaFree`s are plain device
  arrays here, freed by Python's garbage collector.

## Verification

All 34 ports pass `run_all.py` on a Quadro P2000 (sm_61), Python 3.14,
Numba 0.68.1. The code is unchanged for Colab's T4 (sm_75).
