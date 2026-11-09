# Linear Inference (PyTorch) — Results

Challenge: run inference on a `torch.nn.Linear` model.  Given `input`
`[batch, input_size]`, `weight` `[output_size, input_size]` and `bias`
`[output_size]`, compute `output = input @ weight.T + bias` into the provided
`output` tensor using PyTorch's built-in operations.  `1 ≤ batch_size ≤
1000`, `1 ≤ input_size, output_size ≤ 1000`; performance is measured at
`batch_size = 1000`.

This is the one challenge in the collection that is a PyTorch program rather
than a CUDA kernel, so there is no `.cu` file and no Numba port — the
implementation is the PyTorch call sequence itself.

Files:

- `simple_inference.py` — `solve(input, model, output)` and the benchmark
  (renamed from `simple_inf.py`).
- `simple_inference_naive.py` — the naive baseline: one broadcast-dot per
  output column, no GEMM primitive.

## Environment note

The system-wide PyTorch (`2.15.0+cu132`) ships **no kernels for this GPU**:
its lowest supported compute capability is 7.5, while the Quadro P2000 is
6.1 (`cudaErrorNoKernelImageForDevice` on the first operation).  To run and
measure the challenge at all, a separate environment was created with a
cu126 build that still packages Pascal:

```bash
conda create -p /home/d/.venvs/torch126 python=3.12 -y
/home/d/.venvs/torch126/bin/pip install torch==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu126
```

`torch 2.10.0+cu126` reports `cuda available: True`, device
`Quadro P2000`, capability `(6, 1)`, and runs cuBLAS GEMMs correctly (the
cu126 wheels ship sm_60 cubins, which are binary-compatible with sm_61).

Run everything with that interpreter:

```bash
/home/d/.venvs/torch126/bin/python simple_inference.py
```

## Implementation

```python
def solve(input, model, output):
    with torch.no_grad():
        torch.matmul(input, model.weight.t(), out=output)
        output.add_(model.bias)
```

`weight.t()` is a view (no transpose traffic), cuBLAS writes the GEMM
directly into the caller's `output`, and the bias is applied by a cheap
in-place elementwise kernel.  `torch.no_grad()` is appropriate for inference
and is also required for the `out=` form (parameters carry `requires_grad`).

### What was compared (batch = input = output = 1000)

All four built-in formulations produce bit-identical output
(max_abs = 2.3e-05 against a float64 CPU reference, i.e. float32 GEMM
rounding).  Medians of 50–100 repetitions:

| Formulation | Time | TFLOP/s |
|---|---:|---:|
| **`matmul(..., out=out)` + `out.add_(bias)` (shipped)** | **1.22–1.35 ms** | **~1.5** |
| `mm(..., out=out)` + `out.add_(bias)` | 1.29–1.35 ms | ~1.5 |
| `addmm(bias, x, w.t(), out=out)` | 1.48–1.57 ms | ~1.3 |
| `F.linear(x, w, b)` then `out.copy_(...)` | 1.55–1.63 ms | ~1.25 |
| `model(x)` then `out.copy_(...)` | 1.55–1.63 ms | ~1.25 |

The ordering is stable across sessions; absolute times drift by ~10% with
GPU clock state.  Two findings worth recording:

- **`addmm` is not the fastest way to add the bias here.**  Despite being the
  single-call form, it measures ~15% slower than matmul-into-`out` followed
  by an in-place add — its bias-broadcast path evidently costs more than a
  separate 4 MB elementwise kernel.
- Allocating a result (`F.linear`, `model(x)`) and copying it into `output`
  costs ~0.25 ms at this shape (the extra 4 MB write + 4 MB read).

At 2 GFLOP (2·batch·in·out), the shipped form reaches ~1.5 TFLOP/s, about
half of the P2000's ~3 TFLOP/s FP32 peak — a normal figure for a
1000×1000×1000 cuBLAS GEMM on a 6-SM card.

### Naive baseline

`simple_inference_naive.py` computes the same result without any GEMM
primitive, one output column at a time:

```python
for o in range(output_size):
    output[:, o] = (input * weight[o]).sum(dim=1) + bias[o]
```

That launches `output_size` kernels and re-reads the whole input once per
output column — 1000 kernels and ~4 GB of traffic at the benchmark shape
instead of one GEMM and 8 MB.  It measures **~164 ms**, about **122× slower**
than the shipped implementation, with the same output (max_abs 2.1e-06 vs the
float64 reference).

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, 6 SMs) |
| PyTorch | 2.10.0+cu126 (in `/home/d/.venvs/torch126`) |
| CUDA driver | 580.178.04 |
| Python | 3.12 (conda env) |
| Measurement date | 2026-11-09 |

Times are medians after a warm-up, including `torch.cuda.synchronize()`.

## Correctness

Every case is checked against a float64 CPU reference
(`input.double() @ weight.double().T + bias.double()`); the reported figure
is max absolute error and the max error relative to the largest reference
value (a per-element relative error is meaningless where the reference
crosses zero).

| Case | max_abs | rel |
|---|---:|---:|
| challenge example 1 (`[[1,2]]`, weight `[[0.5,1],[1.5,0.5]]`, bias `[0.1,0.2]`) | 9.7e-08 | 3.7e-08 |
| challenge example 2 (3×1, weight `[[2]]`, bias `[1]`) | exact | exact |
| 1×1 → 1 | 5.4e-08 | 1.5e-08 |
| 1×1000 → 1000 | 1.7e-06 | 1.3e-07 |
| 64×64 → 64 | 3.2e-06 | 2.2e-07 |
| 1000×1000 → 1000 (benchmark shape) | 2.2e-05 | 1.3e-06 |
| 1000×512 → 256 | 1.5e-05 | 9.7e-07 |
| 1000×10 → 1000 | 2.4e-06 | 1.2e-07 |
| 1000×1000 → 10 | 3.5e-06 | 2.4e-07 |

The shipped implementation and the naive baseline agree with each other and
with the float64 reference to float32 GEMM precision.

## Performance — batch = input = output = 1000

| Implementation | Time | vs optimized |
|---|---:|---:|
| `simple_inference_naive.py` (one kernel per output column) | ~164 ms | ~122× slower |
| `F.linear` / `model(x)` + copy | 1.55–1.63 ms | 1.15–1.20× slower |
| `addmm(..., out=)` | 1.48–1.57 ms | 1.10–1.15× slower |
| **`simple_inference.py`** (`matmul(out=)` + `add_`) | **1.22–1.35 ms** | 1.0× |

All variants produce identical results; the differences are in how many
kernels run and how much extra memory traffic they add.
