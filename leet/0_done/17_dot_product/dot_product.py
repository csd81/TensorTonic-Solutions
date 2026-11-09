"""Numba CUDA port of dot_product.cu.

Dot product of two float vectors, N = 4,194,304 at the benchmark size, so the
kernel reads 32 MB and is purely bandwidth-bound.  Same shape as the C++
version: a grid-stride loop with per-thread accumulation, a warp/block
reduction through shuffles and shared memory, and one atomicAdd per block.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32

THREADS = 256
NWARPS = THREADS // 32
FULL = 0xFFFFFFFF

_blocks = 1024


@cuda.jit(cache=True)
def zero_kernel(a):
    if cuda.grid(1) == 0:
        a[0] = float32(0.0)


@cuda.jit(cache=True)
def dot_kernel(A, B, result, n):
    shared = cuda.shared.array(shape=NWARPS, dtype=float32)
    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5

    s = float32(0.0)
    i = cuda.blockIdx.x * cuda.blockDim.x + tid
    stride = cuda.blockDim.x * cuda.gridDim.x
    while i < n:
        s += A[i] * B[i]
        i += stride

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
            cuda.atomic.add(result, 0, v)


def solve(A, B, result, N):
    if A is None or B is None or result is None:
        return
    if N <= 0:
        raise ValueError("N must be positive")

    X = A if getattr(A, "ndim", 1) == 1 else A.reshape(N)
    Y = B if getattr(B, "ndim", 1) == 1 else B.reshape(N)

    blocks = min((N + THREADS - 1) // THREADS, _blocks)
    zero_kernel[1, 1](result)
    dot_kernel[blocks, THREADS](X, Y, result, N)
    cuda.synchronize()


def _run(name, a, b):
    n = a.size
    ref = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
    dA = cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    dB = cuda.to_device(np.ascontiguousarray(b, dtype=np.float32))
    dR = cuda.device_array(1, dtype=np.float32)
    solve(dA, dB, dR, n)
    got = float(dR.copy_to_host()[0])
    err = abs(got - ref) / max(1.0, abs(ref))
    ok = err < 1e-4
    print(f"{name:>22} N={n}: got={got:.6f} ref={ref:.6f} rel={err:.2e} "
          f"{'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([1, 2, 3, 4], np.float32),
               np.array([5, 6, 7, 8], np.float32))
    ok &= _run("example 2", np.array([0.5, 1.5, 2.5], np.float32),
               np.array([2, 3, 4], np.float32))
    ok &= _run("single", np.array([7.0], np.float32),
               np.array([3.0], np.float32))
    ok &= _run("cancelling", np.array([1.0, -1.0], np.float32),
               np.array([1.0, 1.0], np.float32))
    for n, name in ((1000, "1k random"), (100000, "100k random"),
                    (4194304, "benchmark shape")):
        ok &= _run(name, rng.uniform(-1000, 1000, n).astype(np.float32),
                   rng.uniform(-1000, 1000, n).astype(np.float32))

    # Performance: N = 4,194,304.
    import statistics
    import time

    N = 4194304
    a = rng.uniform(-1000, 1000, N).astype(np.float32)
    b = rng.uniform(-1000, 1000, N).astype(np.float32)
    dA = cuda.to_device(a)
    dB = cuda.to_device(b)
    dR = cuda.device_array(1, dtype=np.float32)
    solve(dA, dB, dR, N)  # warm-up
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        solve(dA, dB, dR, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    ref = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
    got = float(dR.copy_to_host()[0])
    err = abs(got - ref) / max(1.0, abs(ref))
    print(f"Perf N={N}: {ms:.4f} ms, {2 * N * 4 / (ms * 1e6):.1f} GB/s, "
          f"rel={err:.2e}, {'OK' if err < 1e-4 else 'FAIL'}")
    ok &= err < 1e-4
    print("ALL OK" if ok else "FAILURES")
