"""PyTorch inference for a torch.nn.Linear layer.

Challenge: given `input` [batch, input_size] and a loaded `nn.Linear` model
(weight [output_size, input_size], bias [output_size]), compute
`output = input @ weight.T + bias` into the provided output tensor, using
PyTorch's built-in operations.

Implementation note: three built-in ways to write into the caller's output
tensor were measured at the benchmark shape (batch = input = output = 1000):

    torch.matmul(x, w.t(), out=out); out.add_(bias)   1.22-1.27 ms
    torch.mm(x, w.t(), out=out);     out.add_(bias)   1.29-1.32 ms
    torch.addmm(bias, x, w.t(), out=out)              1.53-1.57 ms
    F.linear(x, w, b) + out.copy_(...)                1.59-1.60 ms

All four give bit-identical results.  The matmul-into-out plus in-place bias
add wins: cuBLAS writes the GEMM straight into the output, and the bias add is
a cheap 4 MB elementwise kernel, while addmm's bias broadcasting path is
measurably slower here and F.linear allocates an extra tensor that must then
be copied.  weight.t() is a view, so no transpose traffic.

Run with a CUDA-capable PyTorch (see README: the default install here has no
sm_61 kernels; a cu126 build in /home/d/.venvs/torch126 works):

    /home/d/.venvs/torch126/bin/python simple_inference.py
"""

import torch


def solve(input: torch.Tensor, model: torch.nn.Module, output: torch.Tensor):
    with torch.no_grad():
        torch.matmul(input, model.weight.t(), out=output)
        output.add_(model.bias)


def _reference(input, model):
    w = model.weight.detach().double().cpu()
    b = model.bias.detach().double().cpu()
    return input.detach().double().cpu() @ w.T + b


def _run(name, input, model, tol=1e-5):
    out = torch.empty(input.shape[0], model.weight.shape[0],
                      device=input.device, dtype=torch.float32)
    solve(input, model, out)
    torch.cuda.synchronize()
    ref = _reference(input, model)
    got = out.double().cpu()
    err = (got - ref).abs()
    abs_err = err.max().item()
    # Global relative error: max error over max |reference|.  A per-element
    # relative error is meaningless where the reference is near zero.
    rel = abs_err / ref.abs().max().clamp_min(1.0).item()
    ok = abs_err < 1e-3 and rel < tol
    print(f"{name:>24}: max_abs={abs_err:.2e} rel={rel:.2e} "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def _bench(fn, reps=50):
    import statistics
    import time

    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU available for PyTorch")

    torch.manual_seed(0)
    dev = "cuda"
    ok = True

    # Challenge example 1.
    m1 = torch.nn.Linear(2, 2, device=dev)
    with torch.no_grad():
        m1.weight.copy_(torch.tensor([[0.5, 1.0], [1.5, 0.5]]))
        m1.bias.copy_(torch.tensor([0.1, 0.2]))
    ok &= _run("example 1", torch.tensor([[1.0, 2.0]], device=dev), m1)

    # Challenge example 2.
    m2 = torch.nn.Linear(1, 1, device=dev)
    with torch.no_grad():
        m2.weight.copy_(torch.tensor([[2.0]]))
        m2.bias.copy_(torch.tensor([1.0]))
    ok &= _run("example 2",
               torch.tensor([[1.0], [2.0], [3.0]], device=dev), m2)

    # Random shapes.
    for batch, isz, osz in [(1, 1, 1), (1, 1000, 1000), (64, 64, 64),
                            (1000, 1000, 1000), (1000, 512, 256),
                            (1000, 10, 1000), (1000, 1000, 10)]:
        m = torch.nn.Linear(isz, osz, device=dev)
        x = torch.empty(batch, isz, device=dev).uniform_(-10.0, 10.0)
        ok &= _run(f"{batch}x{isz}->{osz}", x, m)

    # Benchmark: batch_size = 1000 (the challenge's measurement point),
    # input_size = output_size = 1000.
    B, I, O = 1000, 1000, 1000
    m = torch.nn.Linear(I, O, device=dev)
    x = torch.empty(B, I, device=dev).uniform_(-10.0, 10.0)
    out = torch.empty(B, O, device=dev)

    def f_matmul_add():
        with torch.no_grad():
            torch.matmul(x, m.weight.t(), out=out)
            out.add_(m.bias)

    def f_mm_add():
        with torch.no_grad():
            torch.mm(x, m.weight.t(), out=out)
            out.add_(m.bias)

    def f_addmm():
        with torch.no_grad():
            torch.addmm(m.bias, x, m.weight.t(), out=out)

    def f_model_copy():
        with torch.no_grad():
            out.copy_(m(x))

    def f_linear_copy():
        with torch.no_grad():
            out.copy_(torch.nn.functional.linear(x, m.weight, m.bias))

    for name, fn in (("matmul out= + add_", f_matmul_add),
                     ("mm out= + add_", f_mm_add),
                     ("addmm out=", f_addmm),
                     ("F.linear + copy_", f_linear_copy),
                     ("model(x) + copy_", f_model_copy)):
        ms = _bench(fn)
        flops = 2.0 * B * I * O
        print(f"Perf {name:>20}: {ms:.4f} ms, {flops / (ms * 1e-3) / 1e12:.3f} "
              f"TFLOP/s")

    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from simple_inference_naive import solve as solve_naive

        def f_naive():
            solve_naive(x, m, out)

        ms = _bench(f_naive, reps=5)
        print(f"Perf {'row-wise naive':>20}: {ms:.4f} ms")
        f_naive()
        torch.cuda.synchronize()
        ref = _reference(x, m)
        err = (out.double().cpu() - ref).abs().max().item()
        print(f"  naive check: max_abs={err:.2e} {'OK' if err < 1e-3 else 'FAIL'}")
        ok &= err < 1e-3
    except ImportError:
        print("  (simple_inference_naive.py not found; skipping naive baseline)")

    print("ALL OK" if ok else "FAILURES")
