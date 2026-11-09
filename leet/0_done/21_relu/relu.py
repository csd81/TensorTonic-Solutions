"""Numba CUDA port of relu.cu.

ReLU: out[i] = max(0, in[i]).  At the benchmark size (N = 25,000,000) the
kernel reads 100 MB and writes 100 MB, so it is a pure read+write streaming
problem.  The C++ version uses float4 accesses and measures the same 2.38 ms
as a pure float4 copy of the same traffic -- the starter's scalar
one-element-per-thread form is already at the DRAM wall.

This port keeps that scalar one-element-per-thread mapping: with fully
coalesced accesses the load width does not matter at the wall, and Numba has
no float4 vector type anyway.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32

THREADS = 256


@cuda.jit(cache=True)
def relu_kernel(input, output, n):
    i = cuda.grid(1)
    if i < n:
        v = input[i]
        output[i] = v if v > float32(0.0) else float32(0.0)


def solve(input, output, N):
    """input and output are float32 device arrays of length N."""
    if N <= 0:
        return
    if input is None or output is None:
        return
    blocks = (N + THREADS - 1) // THREADS
    relu_kernel[blocks, THREADS](input, output, N)
    cuda.synchronize()


def _run(name, a):
    n = a.size
    ref = np.maximum(a, np.float32(0.0))
    da = cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    dout = cuda.device_array(n, dtype=np.float32)
    solve(da, dout, n)
    got = dout.copy_to_host()
    bad = int(np.count_nonzero(got != ref))
    ok = bad == 0
    print(f"{name:>22} N={n}: mismatches={bad} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([-2.0, -1.0, 0.0, 1.0, 2.0], np.float32))
    ok &= _run("example 2", np.array([-3.5, 0.0, 4.2], np.float32))
    ok &= _run("single negative", np.array([-7.0], np.float32))
    ok &= _run("single positive", np.array([7.0], np.float32))
    ok &= _run("all negative", np.full(100, -1.5, np.float32))
    ok &= _run("all positive", np.full(100, 2.5, np.float32))
    ok &= _run("all zero", np.zeros(64, np.float32))
    for n in (2, 4, 5, 7, 1023, 1024, 1025, 4099):
        a = rng.uniform(-100.0, 100.0, n).astype(np.float32)
        a[rng.random(n) < 0.5] *= -1.0
        ok &= _run(f"random {n}", a)

    # Benchmark: N = 25,000,000.
    import statistics
    import time

    N = 25000000
    a = rng.uniform(-100.0, 100.0, N).astype(np.float32)
    a[rng.random(N) < 0.5] *= -1.0
    da = cuda.to_device(a)
    dout = cuda.device_array(N, dtype=np.float32)
    solve(da, dout, N)  # warm-up, also triggers the JIT compile
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        solve(da, dout, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    ref = np.maximum(a, np.float32(0.0))
    bad = int(np.count_nonzero(dout.copy_to_host() != ref))
    bytes_moved = 2 * N * 4
    print(f"Perf N={N}: {ms:.4f} ms, {bytes_moved / (ms * 1e6):.1f} GB/s, "
          f"mismatches={bad}, {'OK' if bad == 0 else 'FAIL'}")
    ok &= bad == 0
    print("ALL OK" if ok else "FAILURES")
