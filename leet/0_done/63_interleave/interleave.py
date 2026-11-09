"""Numba CUDA port of interleave.cu.

Interleave A and B (each N floats) into 2N floats:
out[2i] = A[i], out[2i+1] = B[i].

At N = 25,000,000 the kernel moves 400 MB, so it is pure DRAM bandwidth.  In
C++ the scalar one-thread-per-index form (this port's mapping) and vectorized
interleaves all measure 4.8-4.9 ms at the ~82 GB/s wall; the vectorized forms
are actually slightly slower because they add registers without reducing DRAM
traffic.  Numba has no float4 anyway.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda

THREADS = 512


@cuda.jit(cache=True)
def interleave_kernel(a, b, out, n):
    i = cuda.grid(1)
    if i < n:
        out[2 * i] = a[i]
        out[2 * i + 1] = b[i]


def solve(A, B, output, N):
    """A, B are float32 device arrays of length N; output of length 2N."""
    if N <= 0:
        return
    if A is None or B is None or output is None:
        return
    blocks = (N + THREADS - 1) // THREADS
    interleave_kernel[blocks, THREADS](A, B, output, N)
    cuda.synchronize()


def _run(name, a, b):
    n = a.size
    ref = np.empty(2 * n, np.float32)
    ref[0::2] = a
    ref[1::2] = b
    da = cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    db = cuda.to_device(np.ascontiguousarray(b, dtype=np.float32))
    dout = cuda.device_array(2 * n, dtype=np.float32)
    solve(da, db, dout, n)
    got = dout.copy_to_host()
    bad = int(np.count_nonzero(got != ref))
    ok = bad == 0
    print(f"{name:>20} N={n}: mismatches={bad} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([1.0, 2.0, 3.0], np.float32),
               np.array([4.0, 5.0, 6.0], np.float32))
    ok &= _run("example 2", np.array([10.0, 20.0], np.float32),
               np.array([30.0, 40.0], np.float32))
    ok &= _run("N=1", np.array([7.0], np.float32), np.array([3.0], np.float32))
    for n in (2, 3, 5, 7, 1023, 1024, 1025, 4099, 100000):
        ok &= _run(f"random {n}", rng.uniform(-1000, 1000, n).astype(np.float32),
                   rng.uniform(-1000, 1000, n).astype(np.float32))

    # Benchmark: N = 25,000,000.
    import statistics
    import time

    N = 25000000
    a = rng.uniform(-1000, 1000, N).astype(np.float32)
    b = rng.uniform(-1000, 1000, N).astype(np.float32)
    da = cuda.to_device(a)
    db = cuda.to_device(b)
    dout = cuda.device_array(2 * N, dtype=np.float32)
    solve(da, db, dout, N)  # warm-up, also triggers the JIT compile
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        solve(da, db, dout, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    got = dout.copy_to_host()
    bad = int(np.count_nonzero(got[0::2] != a) + np.count_nonzero(got[1::2] != b))
    bytes_moved = 4 * N * 4
    print(f"Perf N={N}: {ms:.4f} ms, {bytes_moved / (ms * 1e6):.1f} GB/s, "
          f"mismatches={bad}, {'OK' if bad == 0 else 'FAIL'}")
    ok &= bad == 0
    print("ALL OK" if ok else "FAILURES")
