"""Numba CUDA port of swiglu.cu.

SWiGLU: split x into halves x1 = x[0..N/2), x2 = x[N/2..N) and compute
out[i] = SiLU(x1[i]) * x2[i] (halfN outputs).

At the benchmark (N = 100,000) the traffic is 600 KB, so the kernel is
launch-bound: an empty launch costs ~6.7 us and a kernel with the same
traffic ~8.0 us in C++.  The C++ version uses float4 and __fdividef to reach
8.5 us; this port has neither (Numba has no float4 type and uses IEEE
division), so it pays the scalar + full-division cost.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import math

import numpy as np
from numba import cuda

THREADS = 256


@cuda.jit(cache=True)
def swiglu_kernel(input, output, half_n):
    i = cuda.grid(1)
    if i < half_n:
        x1 = input[i]
        x2 = input[i + half_n]
        output[i] = (x1 / (1.0 + math.exp(-x1))) * x2


def solve(input, output, N):
    """input is a float32 device array of length N, output of length N//2."""
    if N <= 0:
        return
    if input is None or output is None:
        return
    half_n = N // 2
    if half_n <= 0:
        return
    blocks = (half_n + THREADS - 1) // THREADS
    swiglu_kernel[blocks, THREADS](input, output, half_n)
    cuda.synchronize()


def _run(name, a):
    n = a.size
    half_n = n // 2
    a64 = a.astype(np.float64)
    ref = (a64[:half_n] / (1.0 + np.exp(-a64[:half_n]))) * a64[half_n:]
    da = cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    dout = cuda.device_array(half_n, dtype=np.float32)
    solve(da, dout, n)
    got = dout.copy_to_host().astype(np.float64)
    abs_err = np.abs(got - ref)
    rel = np.max(abs_err / np.maximum(1e-6, np.abs(ref)))
    ok = np.max(abs_err) < 1e-1 and rel < 1e-5
    print(f"{name:>22} N={n}: max_abs={np.max(abs_err):.2e} "
          f"max_rel={rel:.2e} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([1.0, 2.0, 3.0, 4.0], np.float32))
    ok &= _run("example 2", np.array([0.5, 1.0], np.float32))
    ok &= _run("zeros", np.zeros(2, np.float32))
    ok &= _run("extremes", np.array([-100.0, 100.0, 100.0, -100.0], np.float32))
    ok &= _run("all equal", np.full(200, 3.0, np.float32))
    for n in (2, 4, 6, 10, 1024, 1026, 4098, 100000):
        ok &= _run(f"random {n}", rng.uniform(-100.0, 100.0, n).astype(np.float32))

    # Benchmark: N = 100,000.
    import statistics
    import time

    N = 100000
    half_n = N // 2
    a = rng.uniform(-100.0, 100.0, N).astype(np.float32)
    da = cuda.to_device(a)
    dout = cuda.device_array(half_n, dtype=np.float32)
    solve(da, dout, N)  # warm-up, also triggers the JIT compile
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        solve(da, dout, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    a64 = a.astype(np.float64)
    ref = (a64[:half_n] / (1.0 + np.exp(-a64[:half_n]))) * a64[half_n:]
    err = float(np.max(np.abs(dout.copy_to_host().astype(np.float64) - ref)))
    bytes_moved = 3 * half_n * 4
    print(f"Perf N={N}: {ms:.4f} ms, {bytes_moved / (ms * 1e6):.1f} GB/s, "
          f"max_abs={err:.2e}, {'OK' if err < 1e-1 else 'FAIL'}")
    ok &= err < 1e-1
    print("ALL OK" if ok else "FAILURES")
