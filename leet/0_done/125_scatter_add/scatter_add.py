"""Numba CUDA port of scatter_add.cu.

Row-wise scatter-add: output[index[i]][d] += src[i][d], with the output
starting at zero.

In C++ the atomic form (one atomicAdd per element) measures 19.7 ms at
N=1,000,000, D=128, M=50,000; the shipped counting-sort pipeline (bucket rows
by destination, one warp sums each destination row and writes it once)
measures 8.65 ms.  The counting sort needs chunked shared-memory scans and
several kernels, which is not practical in Numba, so this port is the atomic
form -- the C++ naive baseline's shape.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda

THREADS = 256


@cuda.jit(cache=True)
def zero_kernel(a, n):
    i = cuda.grid(1)
    if i < n:
        a[i] = 0.0


@cuda.jit(cache=True)
def scatter_add_kernel(src, index, output, total, D):
    for t in range(cuda.grid(1), total, cuda.gridsize(1)):
        i = t // D
        d = t % D
        cuda.atomic.add(output, index[i] * D + d, src[t])


def solve(src, index, output, N, D, M):
    """src: float32 device array N*D; index: int32 device array N;
    output: float32 device array M*D (contents undefined)."""
    if N <= 0 or D <= 0 or M <= 0:
        return
    total_out = M * D
    zero_kernel[(total_out + THREADS - 1) // THREADS, THREADS](output, total_out)
    total = N * D
    blocks = (total + THREADS - 1) // THREADS
    if blocks > 65535:
        blocks = 65535
    scatter_add_kernel[blocks, THREADS](src, index, output, total, D)
    cuda.synchronize()


def _run(name, src, idx, N, D, M):
    ref = np.zeros((M, D), np.float64)
    np.add.at(ref, idx, src.astype(np.float64))
    dsrc = cuda.to_device(np.ascontiguousarray(src, dtype=np.float32).reshape(-1))
    didx = cuda.to_device(np.ascontiguousarray(idx, dtype=np.int32))
    dout = cuda.device_array(M * D, dtype=np.float32)
    solve(dsrc, didx, dout, N, D, M)
    got = dout.copy_to_host().reshape(M, D).astype(np.float64)
    abs_err = np.abs(got - ref)
    bad = int(np.count_nonzero(abs_err > 2e-2 + 1e-4 * np.abs(ref)))
    ok = bad == 0
    print(f"{name:>18} N={N} D={D} M={M}: max_abs={np.max(abs_err):.2e} "
          f"bad={bad} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example", np.array([[1, 2], [3, 4], [5, 6], [7, 8]], np.float32),
               np.array([0, 2, 0, 1], np.int32), 4, 2, 3)
    for N, D, M in ((1, 1, 1), (10, 3, 2), (1000, 33, 100), (5000, 128, 50),
                    (100000, 17, 1000), (100000, 256, 100000)):
        src = rng.uniform(-10.0, 10.0, (N, D)).astype(np.float32)
        idx = rng.integers(0, M, N, dtype=np.int32)
        ok &= _run(f"random {N}x{D}->{M}", src, idx, N, D, M)
    src = rng.uniform(-10.0, 10.0, (50000, 64)).astype(np.float32)
    ok &= _run("all-to-one", src, np.zeros(50000, np.int32), 50000, 64, 1000)

    # Benchmark: N = 1,000,000, D = 128, M = 50,000.
    import statistics
    import time

    N, D, M = 1000000, 128, 50000
    src = rng.uniform(-10.0, 10.0, (N, D)).astype(np.float32)
    idx = rng.integers(0, M, N, dtype=np.int32)
    dsrc = cuda.to_device(src.reshape(-1))
    didx = cuda.to_device(idx)
    dout = cuda.device_array(M * D, dtype=np.float32)
    solve(dsrc, didx, dout, N, D, M)  # warm-up, also triggers the JIT compile
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(dsrc, didx, dout, N, D, M)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    ref = np.zeros((M, D), np.float64)
    np.add.at(ref, idx, src.astype(np.float64))
    got = dout.copy_to_host().reshape(M, D).astype(np.float64)
    err = float(np.max(np.abs(got - ref)))
    bytes_moved = (N * D + M * D) * 4 + N * 4
    print(f"Perf N={N} D={D} M={M}: {ms:.4f} ms, "
          f"{bytes_moved / (ms * 1e6):.1f} GB/s, max_abs={err:.2e}, "
          f"{'OK' if err < 3 else 'FAIL'}")
    ok &= err < 3
    print("ALL OK" if ok else "FAILURES")
