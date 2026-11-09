"""Numba CUDA port of gelu.cu.

Exact (erf-based) GELU: out[i] = 0.5*x*(1 + erf(x/sqrt(2))).

At the benchmark (N = 50,000,000) the kernel moves 400 MB, so it is
memory-bound: in C++ the scalar and float4 forms both measure 4.85-4.89 ms,
the same as a pure float4 copy (4.85-4.88 ms).  This port keeps the scalar
mapping (Numba has no float4 type).

`math.erf` computes in float64 in Numba (see the GEGLU port); this uses
`numba.cuda.libdevice.erff`, the float32 erf, with float32 constants.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import math

import numpy as np
from numba import cuda, float32
from numba.cuda import libdevice

THREADS = 256
HALF = float32(0.5)
ONE = float32(1.0)
INV_SQRT_2 = float32(0.7071067811865475)


@cuda.jit(cache=True)
def gelu_kernel(X, Y, n):
    i = cuda.grid(1)
    if i < n:
        x = X[i]
        Y[i] = HALF * x * (ONE + libdevice.erff(x * INV_SQRT_2))


def solve(X, Y, N):
    """X and Y are float32 device arrays of length N."""
    if N <= 0:
        return
    if X is None or Y is None:
        return
    blocks = (N + THREADS - 1) // THREADS
    gelu_kernel[blocks, THREADS](X, Y, N)
    cuda.synchronize()


def _ref(a):
    a64 = a.astype(np.float64)
    return 0.5 * a64 * (1.0 + np.vectorize(math.erf)(a64 / math.sqrt(2.0)))


def _run(name, a):
    n = a.size
    ref = _ref(a)
    da = cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    dout = cuda.device_array(n, dtype=np.float32)
    solve(da, dout, n)
    got = dout.copy_to_host().astype(np.float64)
    abs_err = np.abs(got - ref)
    bad = int(np.count_nonzero(abs_err > 1e-4 + 1e-5 * np.abs(ref)))
    ok = bad == 0
    print(f"{name:>20} N={n}: max_abs={np.max(abs_err):.2e} bad={bad} "
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
    ref = _ref(a)
    got = dout.copy_to_host().astype(np.float64)
    bad = int(np.count_nonzero(np.abs(got - ref) > 1e-4 + 1e-5 * np.abs(ref)))
    print(f"Perf N={N}: {ms:.4f} ms, {2 * N * 4 / (ms * 1e6):.1f} GB/s, "
          f"max_abs={np.max(np.abs(got - ref)):.2e}, "
          f"{'OK' if bad == 0 else 'FAIL'}")
    ok &= bad == 0
    print("ALL OK" if ok else "FAILURES")
