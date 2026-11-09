"""Numba CUDA port of silu.cu.

SiLU: out[i] = x / (1 + exp(-x)).

At the benchmark size (N = 50,000) the traffic is only 400 KB, so the kernel
is launch-bound: an empty launch costs ~6.7 us and a float4 copy of the same
data ~7.7 us in C++.  The C++ version uses __expf + __fdividef to keep the
math off the critical path and measures 8.2 us; the plain expf + IEEE
division form measures 10.8-12.1 us.

Numba's math.exp maps to expf and its division is IEEE, so this port has the
plain-form cost (measured below); there is no float4 type either.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import math

import numpy as np
from numba import cuda

THREADS = 256


@cuda.jit(cache=True)
def silu_kernel(input, output, n):
    i = cuda.grid(1)
    if i < n:
        x = input[i]
        output[i] = x / (1.0 + math.exp(-x))


def solve(input, output, N):
    """input and output are float32 device arrays of length N."""
    if N <= 0:
        return
    if input is None or output is None:
        return
    blocks = (N + THREADS - 1) // THREADS
    silu_kernel[blocks, THREADS](input, output, N)
    cuda.synchronize()


def _run(name, a):
    n = a.size
    ref = a.astype(np.float64) / (1.0 + np.exp(-a.astype(np.float64)))
    da = cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    dout = cuda.device_array(n, dtype=np.float32)
    solve(da, dout, n)
    got = dout.copy_to_host().astype(np.float64)
    abs_err = np.abs(got - ref)
    rel = np.max(abs_err / np.maximum(1e-6, np.abs(ref)))
    ok = np.max(abs_err) < 1e-4 and rel < 1e-5
    print(f"{name:>22} N={n}: max_abs={np.max(abs_err):.2e} "
          f"max_rel={rel:.2e} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([0.5, 1.0, -0.5], np.float32))
    ok &= _run("example 2", np.array([-1.0, -2.0, -3.0, -4.0, -5.0], np.float32))
    ok &= _run("single zero", np.array([0.0], np.float32))
    ok &= _run("extremes", np.array([-100.0, -50.0, -10.0, 10.0, 50.0, 100.0],
                                    np.float32))
    ok &= _run("all positive", np.full(100, 3.0, np.float32))
    ok &= _run("all negative", np.full(100, -3.0, np.float32))
    for n in (1, 2, 3, 5, 7, 1023, 1024, 1025, 4099, 50000):
        ok &= _run(f"random {n}", rng.uniform(-100.0, 100.0, n).astype(np.float32))

    # Benchmark: N = 50,000.
    import statistics
    import time

    N = 50000
    a = rng.uniform(-100.0, 100.0, N).astype(np.float32)
    da = cuda.to_device(a)
    dout = cuda.device_array(N, dtype=np.float32)
    solve(da, dout, N)  # warm-up, also triggers the JIT compile
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        solve(da, dout, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    ref = a.astype(np.float64) / (1.0 + np.exp(-a.astype(np.float64)))
    got = dout.copy_to_host().astype(np.float64)
    ok &= bool(np.max(np.abs(got - ref)) < 1e-4)
    print(f"Perf N={N}: {ms:.4f} ms, {2 * N * 4 / (ms * 1e6):.1f} GB/s, "
          f"max_abs={np.max(np.abs(got - ref)):.2e}, "
          f"{'OK' if ok else 'FAIL'}")
    print("ALL OK" if ok else "FAILURES")
