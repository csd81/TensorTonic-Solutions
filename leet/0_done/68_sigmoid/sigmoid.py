"""Numba CUDA port of sigmoid.cu.

Sigmoid: out[i] = 1 / (1 + exp(-x)).

At the benchmark (N = 50,000,000) the kernel moves 400 MB, so it is
memory-bound: a float4 copy measures 4.84-5.00 ms (80-83 GB/s).  In C++ the
starter's scalar expf + IEEE division measures 5.06-5.17 ms; the shipped
float4 + __expf + __fdividef kernel measures 4.84-4.92 ms, on the copy floor.

Numba has no float4 type and no fast intrinsics, so this port pays the
scalar + full-precision cost (measured below).  One trap: `math.exp` computes
in float64 in Numba (inspect_types shows a float64 result), and this GPU's
FP64 throughput is 1/32 of FP32 — the first port measured 12.5 ms because of
it.  `numba.cuda.libdevice.expf` is the float32 exponential, and
`libdevice.fast_expf` / `libdevice.fast_fdividef` are the __expf/__fdividef
equivalents used below (worst-case ~1e-6 relative error).

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import math

import numpy as np
from numba import cuda, float32
from numba.cuda import libdevice

THREADS = 256
ONE = float32(1.0)


@cuda.jit(cache=True)
def sigmoid_kernel(X, Y, n):
    i = cuda.grid(1)
    if i < n:
        Y[i] = libdevice.fast_fdividef(ONE, ONE + libdevice.fast_expf(-X[i]))


def solve(X, Y, N):
    """X and Y are float32 device arrays of length N."""
    if N <= 0:
        return
    if X is None or Y is None:
        return
    blocks = (N + THREADS - 1) // THREADS
    sigmoid_kernel[blocks, THREADS](X, Y, N)
    cuda.synchronize()


def _run(name, a):
    n = a.size
    ref = 1.0 / (1.0 + np.exp(-a.astype(np.float64)))
    da = cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    dout = cuda.device_array(n, dtype=np.float32)
    solve(da, dout, n)
    got = dout.copy_to_host().astype(np.float64)
    abs_err = np.abs(got - ref)
    bad = int(np.count_nonzero(abs_err >= 1e-5))
    ok = bad == 0
    print(f"{name:>22} N={n}: max_abs={np.max(abs_err):.2e} bad={bad} "
          f"{'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([0.0, 1.0, -1.0, 2.0], np.float32))
    ok &= _run("example 2", np.array([0.5, -0.5, 3.0, -3.0], np.float32))
    ok &= _run("single zero", np.array([0.0], np.float32))
    ok &= _run("extremes", np.array([-100.0, -50.0, -10.0, 10.0, 50.0, 100.0],
                                    np.float32))
    ok &= _run("tiny values", np.array([1e-6, -1e-6, 1e-30, -1e-30], np.float32))
    for n in (1, 2, 3, 5, 7, 1023, 1024, 1025, 4099, 100000):
        ok &= _run(f"random {n}", rng.uniform(-100.0, 100.0, n).astype(np.float32))

    # Benchmark: N = 50,000,000.
    import statistics
    import time

    N = 50000000
    a = rng.uniform(-100.0, 100.0, N).astype(np.float32)
    da = cuda.to_device(a)
    dout = cuda.device_array(N, dtype=np.float32)
    solve(da, dout, N)  # warm-up, also triggers the JIT compile
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        solve(da, dout, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    ref = 1.0 / (1.0 + np.exp(-a.astype(np.float64)))
    err = float(np.max(np.abs(dout.copy_to_host().astype(np.float64) - ref)))
    bytes_moved = 2 * N * 4
    print(f"Perf N={N}: {ms:.4f} ms, {bytes_moved / (ms * 1e6):.1f} GB/s, "
          f"max_abs={err:.2e}, {'OK' if err < 1e-5 else 'FAIL'}")
    ok &= err < 1e-5
    print("ALL OK" if ok else "FAILURES")
