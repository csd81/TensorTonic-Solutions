"""Numba CUDA port of matrix_copy.cu.

Copy an N x N float matrix: B[i] = A[i].

At N = 4096 the kernel moves 128 MB, so it is a pure DRAM copy.  In C++ the
scalar one-element-per-thread form and a float4 form both measure ~1.63 ms
(82 GB/s) -- the read+write ceiling -- so Numba's lack of a float4 type costs
nothing here.  This port keeps the scalar mapping.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda

THREADS = 256


@cuda.jit(cache=True)
def copy_kernel(a, b, total):
    i = cuda.grid(1)
    if i < total:
        b[i] = a[i]


def solve(A, B, N):
    """A and B are float32 device arrays of length N*N."""
    if N <= 0:
        return
    if A is None or B is None:
        return
    total = N * N
    blocks = (total + THREADS - 1) // THREADS
    copy_kernel[blocks, THREADS](A, B, total)
    cuda.synchronize()


def _run(name, a, N):
    total = N * N
    da = cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    db = cuda.device_array(total, dtype=np.float32)
    solve(da, db, N)
    got = db.copy_to_host()
    bad = int(np.count_nonzero(got != a))
    ok = bad == 0
    print(f"{name:>20} N={N}: mismatches={bad} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([1.0, 2.0, 3.0, 4.0], np.float32), 2)
    ok &= _run("example 2", np.array(
        [5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.2, 12.3, 13.4], np.float32), 3)
    ok &= _run("N=1", np.array([7.0], np.float32), 1)
    for N in (2, 5, 7, 31, 64, 100, 255, 1000, 4096):
        a = rng.uniform(-1000.0, 1000.0, N * N).astype(np.float32)
        ok &= _run(f"random {N}", a, N)

    # Benchmark: N = 4096.
    import statistics
    import time

    N = 4096
    a = rng.uniform(-1000.0, 1000.0, N * N).astype(np.float32)
    da = cuda.to_device(a)
    db = cuda.device_array(N * N, dtype=np.float32)
    solve(da, db, N)  # warm-up, also triggers the JIT compile
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        solve(da, db, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    bad = int(np.count_nonzero(db.copy_to_host() != a))
    bytes_moved = 2 * N * N * 4
    print(f"Perf N={N}: {ms:.4f} ms, {bytes_moved / (ms * 1e6):.1f} GB/s, "
          f"mismatches={bad}, {'OK' if bad == 0 else 'FAIL'}")
    ok &= bad == 0
    print("ALL OK" if ok else "FAILURES")
