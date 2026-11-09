"""Numba CUDA port of value_clipping.cu.

Clip each element of a float32 vector to [lo, hi]: out[i] = min(max(x, lo), hi).

At the benchmark size (N = 100,000) the traffic is only 800 KB, so the kernel
is launch-bound rather than bandwidth-bound: in C++ an empty kernel costs
6.7 us, a float4 copy of the same data 8.6 us, and the float4 clip kernel
9.0 us.  This port uses the plain scalar one-element-per-thread mapping
(Numba has no float4 type), which measures 10.9 us in C++ -- the extra ~2 us
over the vectorized form is the only cost of scalar accesses at this size.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32

THREADS = 256


@cuda.jit(cache=True)
def clip_kernel(input, output, lo, hi, n):
    i = cuda.grid(1)
    if i < n:
        v = input[i]
        if v < lo:
            v = lo
        if v > hi:
            v = hi
        output[i] = v


@cuda.jit(cache=True)
def noop_kernel():
    pass


def solve(input, output, lo, hi, N):
    """input and output are float32 device arrays of length N; lo and hi are
    Python floats (converted to float32 for the kernel)."""
    if N <= 0:
        return
    if input is None or output is None:
        return
    blocks = (N + THREADS - 1) // THREADS
    clip_kernel[blocks, THREADS](input, output, float32(lo), float32(hi), N)
    cuda.synchronize()


def _run(name, a, lo, hi):
    n = a.size
    ref = np.clip(a, np.float32(lo), np.float32(hi))
    da = cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    dout = cuda.device_array(n, dtype=np.float32)
    solve(da, dout, lo, hi, n)
    got = dout.copy_to_host()
    bad = int(np.count_nonzero(got != ref))
    ok = bad == 0
    print(f"{name:>24} N={n} lo={lo} hi={hi}: mismatches={bad} "
          f"{'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([1.5, -2.0, 3.0, 4.5], np.float32),
               0.0, 3.5)
    ok &= _run("example 2", np.array([-1.0, 2.0, 5.0], np.float32),
               -0.5, 2.5)
    ok &= _run("single, clipped high", np.array([7.0], np.float32), 0.0, 1.0)
    ok &= _run("single, clipped low", np.array([-7.0], np.float32), 0.0, 1.0)
    ok &= _run("single, inside", np.array([0.5], np.float32), 0.0, 1.0)
    ok &= _run("all inside", np.full(100, 0.5, np.float32), 0.0, 1.0)
    ok &= _run("all below", np.full(100, -5.0, np.float32), 0.0, 1.0)
    ok &= _run("all above", np.full(100, 5.0, np.float32), 0.0, 1.0)
    ok &= _run("lo == hi", np.full(100, 3.0, np.float32), 3.0, 3.0)
    ok &= _run("full range", np.array([-1e6, 0.0, 1e6], np.float32),
               -1e6, 1e6)
    ok &= _run("crossing zero",
               np.array([-5.0, -1.0, 0.0, 1.0, 5.0], np.float32), -2.0, 2.0)
    for n in (2, 3, 5, 1023, 1025, 99999, 100000):
        ok &= _run(f"random {n}", rng.uniform(-1e6, 1e6, n).astype(np.float32),
                   -1234.5, 6789.0)

    # Benchmark: N = 100,000.
    import statistics
    import time

    N = 100000
    a = rng.uniform(-1e6, 1e6, N).astype(np.float32)
    lo, hi = -12345.0, 54321.0
    da = cuda.to_device(a)
    dout = cuda.device_array(N, dtype=np.float32)
    solve(da, dout, lo, hi, N)  # warm-up, also triggers the JIT compile

    def timeit(fn, reps=30):
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
        return statistics.median(times)

    noop_ms = timeit(lambda: (noop_kernel[1, 1](), cuda.synchronize()))
    ms = timeit(lambda: solve(da, dout, lo, hi, N))
    ref = np.clip(a, np.float32(lo), np.float32(hi))
    bad = int(np.count_nonzero(dout.copy_to_host() != ref))
    bytes_moved = 2 * N * 4
    print(f"Perf N={N}: {ms:.4f} ms, {bytes_moved / (ms * 1e6):.1f} GB/s, "
          f"mismatches={bad}, {'OK' if bad == 0 else 'FAIL'}")
    print(f"  empty Numba kernel launch+sync: {noop_ms:.4f} ms")
    ok &= bad == 0
    print("ALL OK" if ok else "FAILURES")
