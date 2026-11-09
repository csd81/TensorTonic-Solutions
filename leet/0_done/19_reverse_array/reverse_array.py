"""Numba CUDA port of reverse_array.cu.

In-place reversal of a float array.  Every element is read and written once,
so the kernel is purely DRAM-bound; the port keeps the scalar (i, N-1-i)
pair swap, which is what the C++ version settled on after measuring float4
vectors and streaming stores (all within noise, scalar marginally best).

solve(input, N) keeps the challenge signature; `input` may be a Numba device
array (preferred) or a host NumPy array.
"""

import numpy as np
from numba import cuda

THREADS = 512


@cuda.jit
def reverse_kernel(a, n):
    i = cuda.grid(1)
    if i < n // 2:
        k = n - 1 - i
        t = a[i]
        a[i] = a[k]
        a[k] = t


def solve(input, N):
    if input is None:
        return
    if N <= 1:
        return
    A = input if getattr(input, "ndim", 1) == 1 else input.reshape(N)
    half = N // 2
    blocks = (half + THREADS - 1) // THREADS
    reverse_kernel[blocks, THREADS](A, N)
    cuda.synchronize()


def _run(name, arr):
    n = arr.size
    ref = arr[::-1].copy()
    dA = cuda.to_device(np.ascontiguousarray(arr, dtype=np.float32))
    solve(dA, n)
    got = dA.copy_to_host()
    ok = np.array_equal(got, ref)
    print(f"{name:>22} N={n}: {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([1.0, 2.0, 3.0, 4.0], np.float32))
    ok &= _run("example 2", np.array([1.5, 2.5, 3.5], np.float32))
    ok &= _run("single", np.array([7.0], np.float32))
    ok &= _run("two", np.array([1.0, 2.0], np.float32))
    ok &= _run("len 7 equal", np.full(7, 3.0, np.float32))
    for n in (1000, 1001, 255, 256, 257):
        ok &= _run(f"len {n}", np.arange(n, dtype=np.float32))
    ok &= _run("1M random", rng.uniform(-1000, 1000, 1 << 20).astype(np.float32))

    # Performance: N = 25,000,000.
    import statistics
    import time

    N = 25000000
    arr = rng.uniform(-1000, 1000, N).astype(np.float32)
    dA = cuda.to_device(arr)
    solve(dA, N)  # warm-up
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        solve(dA, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    out = dA.copy_to_host()
    good = bool(np.array_equal(out, arr[::-1]))
    print(f"Perf N={N}: {ms:.3f} ms, "
          f"{2.0 * N * 4 / (ms * 1e6):.1f} GB/s, check={'OK' if good else 'FAIL'}")
    ok &= good
    print("ALL OK" if ok else "FAILURES")
