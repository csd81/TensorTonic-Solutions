"""Numba CUDA port of geglu.cu.

GEGLU: split x into halves x1 = x[0..N/2), x2 = x[N/2..N) and compute
out[i] = x1[i] * GELU(x2[i]), GELU(y) = 0.5*y*(1 + erf(y/sqrt(2))).

At the benchmark (N = 1,000,000) the kernel moves 6 MB, so it is memory-bound:
in C++ the scalar starter, the naive form and the float4 version all measure
0.0784-0.0789 ms (76 GB/s) — the erff is hidden behind the loads.  This port
keeps the scalar mapping (Numba has no float4 type).

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import math

import numpy as np
from numba import cuda, float32
from numba.cuda import libdevice

THREADS = 256
INV_SQRT_2 = float32(0.7071067811865475)
HALF = float32(0.5)
ONE = float32(1.0)


@cuda.jit(cache=True)
def geglu_kernel(input, output, half_n):
    i = cuda.grid(1)
    if i < half_n:
        x1 = input[i]
        x2 = input[i + half_n]
        # libdevice.erff is the float32 erf; math.erf computes in float64,
        # which is ~32x slower on this GPU and dominated the runtime.
        output[i] = x1 * (HALF * x2 * (ONE + libdevice.erff(x2 * INV_SQRT_2)))


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
    geglu_kernel[blocks, THREADS](input, output, half_n)
    cuda.synchronize()


def _ref(a):
    half = a.size // 2
    x1 = a[:half].astype(np.float64)
    x2 = a[half:].astype(np.float64)
    return x1 * (0.5 * x2 * (1.0 + np.vectorize(math.erf)(x2 / math.sqrt(2.0))))


def _run(name, a):
    n = a.size
    ref = _ref(a)
    da = cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    dout = cuda.device_array(n // 2, dtype=np.float32)
    solve(da, dout, n)
    got = dout.copy_to_host().astype(np.float64)
    abs_err = np.abs(got - ref)
    rel = np.max(abs_err) / max(1.0, np.max(np.abs(ref)))
    ok = np.max(abs_err) < 1e-1 and rel < 1e-5
    print(f"{name:>20} N={n}: max_abs={np.max(abs_err):.2e} "
          f"rel={rel:.2e} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([1.0, 1.0], np.float32))
    ok &= _run("example 2", np.array([2.0, -1.0, 1.0, 0.5], np.float32))
    ok &= _run("zeros", np.zeros(2, np.float32))
    ok &= _run("extremes", np.array([-100.0, 100.0, 100.0, -100.0], np.float32))
    ok &= _run("all equal", np.full(200, 3.0, np.float32))
    for n in (2, 4, 6, 10, 1024, 1026, 4098):
        ok &= _run(f"random {n}", rng.uniform(-100.0, 100.0, n).astype(np.float32))

    # Benchmark: N = 1,000,000.
    import statistics
    import time

    N = 1000000
    a = rng.uniform(-100.0, 100.0, N).astype(np.float32)
    da = cuda.to_device(a)
    dout = cuda.device_array(N // 2, dtype=np.float32)
    solve(da, dout, N)  # warm-up, also triggers the JIT compile
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        solve(da, dout, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    ref = _ref(a)
    err = float(np.max(np.abs(dout.copy_to_host().astype(np.float64) - ref)))
    bytes_moved = 3 * (N // 2) * 4
    print(f"Perf N={N}: {ms:.4f} ms, {bytes_moved / (ms * 1e6):.1f} GB/s, "
          f"max_abs={err:.2e}, {'OK' if err < 1e-1 else 'FAIL'}")
    ok &= err < 1e-1
    print("ALL OK" if ok else "FAILURES")
