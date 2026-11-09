"""Numba CUDA port of sparse_matrix_vector_multiplication.cu.

Sparse matrix-vector product with the matrix stored densely in row-major
order (60-70% of the entries are zero, but all of them are in memory).

The benchmark shape is M = 1,000 x N = 10,000: the matrix is 40 MB, so the
kernel is a pure streaming problem and the ceiling is the memory read
bandwidth.  The C++ version maps one 256-thread block to each row, walks the
row with float4 loads, and hits the read ceiling (~86 GB/s).

Numba has no float4 load type, so this port uses the same one-block-per-row
mapping with scalar loads: consecutive threads read consecutive elements, so
the accesses are still fully coalesced.  The inner loop is branchless (zeros
contribute a * 0, which is exact) because the C++ sweep showed the
per-element zero checks cost more than the L1 lookups they save.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32

THREADS = 256
NWARPS = THREADS // 32
FULL = 0xFFFFFFFF


@cuda.jit(cache=True)
def spmv_kernel(A, x, y, N):
    shared = cuda.shared.array(shape=NWARPS, dtype=float32)
    row = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    base = row * N

    s = float32(0.0)
    j = tid
    while j < N:
        s += A[base + j] * x[j]
        j += THREADS

    for off in (16, 8, 4, 2, 1):
        s += cuda.shfl_down_sync(FULL, s, off)
    if lane == 0:
        shared[warp] = s
    cuda.syncthreads()

    if warp == 0:
        v = shared[lane] if lane < NWARPS else float32(0.0)
        for off in (16, 8, 4, 2, 1):
            v += cuda.shfl_down_sync(FULL, v, off)
        if lane == 0:
            y[row] = v


def solve(A, x, y, M, N, nnz=0):
    """A is a flat length-M*N float32 device array holding the row-major
    matrix, x is length N, y is length M.  nnz is accepted for signature
    parity with the challenge and is not needed: A is stored densely, so every
    entry must be read."""
    if M <= 0 or N <= 0:
        return
    spmv_kernel[M, THREADS](A, x, y, N)
    cuda.synchronize()


def _run(name, a, x):
    M, N = a.shape
    ref = a.astype(np.float64) @ x.astype(np.float64)
    dA = cuda.to_device(np.ascontiguousarray(a, dtype=np.float32).reshape(-1))
    dx = cuda.to_device(np.ascontiguousarray(x, dtype=np.float32))
    dy = cuda.device_array(M, dtype=np.float32)
    solve(dA, dx, dy, M, N)
    got = dy.copy_to_host()
    err = float(np.max(np.abs(got.astype(np.float64) - ref))) / max(
        1.0, float(np.max(np.abs(ref))))
    ok = err < 1e-5
    print(f"{name:>22} M={M} N={N}: max_err={err:.2e} "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def _sparse(rng, M, N, zero_pct=65):
    a = rng.uniform(-10.0, 10.0, (M, N)).astype(np.float32)
    a[rng.random((M, N)) < zero_pct / 100.0] = 0.0
    x = rng.uniform(-10.0, 10.0, N).astype(np.float32)
    return a, x


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example", np.array([[5, 0, 0, 1], [0, 2, 3, 0], [0, 0, 0, 4]],
                                   np.float32),
               np.array([1, 2, 3, 4], np.float32))
    ok &= _run("1x1", np.array([[7.0]], np.float32),
               np.array([3.0], np.float32))
    ok &= _run("all zero", np.zeros((2, 2), np.float32),
               np.array([1, 2], np.float32))
    ok &= _run("all ones", np.ones((2, 2), np.float32),
               np.array([2, 3], np.float32))
    ok &= _run("N=2", np.array([[1, 2], [3, 4], [5, 6]], np.float32),
               np.array([1, 1], np.float32))
    ok &= _run("1x1000", *_sparse(rng, 1, 1000))
    ok &= _run("999x1000", *_sparse(rng, 999, 1000))
    for N in (1001, 1002, 1003):
        ok &= _run(f"37x{N}", *_sparse(rng, 37, N))
    ok &= _run("999x9999", *_sparse(rng, 999, 9999))

    # Benchmark: M = 1,000 x N = 10,000, ~65% zeros.
    import statistics
    import time

    M, N = 1000, 10000
    a, x = _sparse(rng, M, N)
    dA = cuda.to_device(a.reshape(-1))
    dx = cuda.to_device(x)
    dy = cuda.device_array(M, dtype=np.float32)
    solve(dA, dx, dy, M, N)  # warm-up, also triggers the JIT compile
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        solve(dA, dx, dy, M, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    ref = a.astype(np.float64) @ x.astype(np.float64)
    got = dy.copy_to_host()
    err = float(np.max(np.abs(got.astype(np.float64) - ref))) / max(
        1.0, float(np.max(np.abs(ref))))
    bytes_read = M * N * 4
    print(f"Perf M={M} N={N}: {ms:.4f} ms, {bytes_read / (ms * 1e6):.1f} GB/s, "
          f"rel={err:.2e}, {'OK' if err < 1e-5 else 'FAIL'}")
    ok &= err < 1e-5
    print("ALL OK" if ok else "FAILURES")
