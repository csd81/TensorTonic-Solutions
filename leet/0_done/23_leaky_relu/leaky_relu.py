"""Numba CUDA port of leaky_relu.cu.

Leaky ReLU with alpha = 0.01: out[i] = x > 0 ? x : 0.01 * x.

At the benchmark size (N = 50,000,000) the kernel reads 200 MB and writes
200 MB, so it is a pure read+write streaming problem.  The C++ version uses
float4 accesses and measures 4.87 ms, the same as a pure float4 copy of the
same traffic (~4.86 ms) -- the starter's shape is already at the DRAM wall.

This port keeps the scalar one-element-per-thread mapping: fully coalesced
scalar accesses saturate DRAM just as well at this size, and Numba has no
float4 type.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32

THREADS = 256
ALPHA = float32(0.01)


@cuda.jit(cache=True)
def leaky_relu_kernel(input, output, n):
    i = cuda.grid(1)
    if i < n:
        x = input[i]
        output[i] = x if x > float32(0.0) else ALPHA * x


def solve(input, output, N):
    """input and output are float32 device arrays of length N."""
    if N <= 0:
        return
    if input is None or output is None:
        return
    blocks = (N + THREADS - 1) // THREADS
    leaky_relu_kernel[blocks, THREADS](input, output, N)
    cuda.synchronize()


def _run(name, a):
    n = a.size
    ref = np.where(a > 0.0, a, np.float32(0.01) * a).astype(np.float32)
    da = cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    dout = cuda.device_array(n, dtype=np.float32)
    solve(da, dout, n)
    got = dout.copy_to_host()
    bad = int(np.count_nonzero(got != ref))
    ok = bad == 0
    print(f"{name:>24} N={n}: mismatches={bad} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([1.0, -2.0, 3.0, -4.0], np.float32))
    ok &= _run("example 2", np.array([-1.5, 0.0, 2.5, -3.0], np.float32))
    ok &= _run("single positive", np.array([7.0], np.float32))
    ok &= _run("single negative", np.array([-7.0], np.float32))
    ok &= _run("single zero", np.array([0.0], np.float32))
    ok &= _run("all negative", np.full(100, -2.0, np.float32))
    ok &= _run("all positive", np.full(100, 2.0, np.float32))
    ok &= _run("all zero", np.zeros(64, np.float32))
    for n in (2, 3, 4, 5, 7, 1023, 1024, 1025, 4099, 100000):
        ok &= _run(f"random {n}", rng.uniform(-1000.0, 1000.0, n).astype(np.float32))

    # Benchmark: N = 50,000,000.
    import statistics
    import time

    N = 50000000
    a = rng.uniform(-1000.0, 1000.0, N).astype(np.float32)
    da = cuda.to_device(a)
    dout = cuda.device_array(N, dtype=np.float32)
    solve(da, dout, N)  # warm-up, also triggers the JIT compile
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        solve(da, dout, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    ref = np.where(a > 0.0, a, np.float32(0.01) * a).astype(np.float32)
    bad = int(np.count_nonzero(dout.copy_to_host() != ref))
    bytes_moved = 2 * N * 4
    print(f"Perf N={N}: {ms:.4f} ms, {bytes_moved / (ms * 1e6):.1f} GB/s, "
          f"mismatches={bad}, {'OK' if bad == 0 else 'FAIL'}")
    ok &= bad == 0
    print("ALL OK" if ok else "FAILURES")
