"""Numba CUDA port of gemm.cu.

FP16 GEMM with FP32 accumulation: C = alpha*(A@B) + beta*C.

At 1024 x 1024 x 1024 this is 2.15 GFLOP on 6 MB — compute-bound.  In C++ the
starter's 32x32-tile, one-output-per-thread kernel measures 0.28 TFLOP/s
(shared-memory-bandwidth-bound: two shared loads per FMA); the shipped
register-blocked kernel reaches 1.04 TFLOP/s.  WMMA is unavailable on this
GPU (compute capability 6.1 has no tensor cores).

Numba can express the same tiled structure (shared arrays are fixed-shape
2-D arrays; each thread keeps one FP32 accumulator), but not the 8x8 register
tile, because Numba has no register-array abstraction that would let a thread
hold 64 accumulators as a value array without local memory.  This port is
therefore the starter's tiled shape written in Numba, for comparison.

Numba cannot index `float16` arrays in CUDA kernels (nor convert float16
scalars), so the matrices are passed as `int16` views and converted with
`libdevice.half2float` / `libdevice.float2half_rn` -- the same bit patterns,
so the port remains a true FP16-in / FP32-accumulate / FP16-out kernel.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import math

import numpy as np
from numba import cuda, float32, int16
from numba.cuda import libdevice

TILE = 32
TPB = TILE * TILE


@cuda.jit(cache=True)
def gemm_kernel(A, B, C, M, N, K, alpha, beta):
    As = cuda.shared.array((TILE, TILE), dtype=float32)
    Bs = cuda.shared.array((TILE, TILE), dtype=float32)
    row = cuda.blockIdx.y * TILE + cuda.threadIdx.y
    col = cuda.blockIdx.x * TILE + cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tx = cuda.threadIdx.x
    acc = float32(0.0)

    num_tiles = (K + TILE - 1) // TILE
    for t in range(num_tiles):
        a_col = t * TILE + tx
        b_row = t * TILE + ty
        a_raw = A[row * K + a_col] if (row < M and a_col < K) else int16(0)
        b_raw = B[b_row * N + col] if (b_row < K and col < N) else int16(0)
        As[ty, tx] = libdevice.half2float(a_raw)
        Bs[ty, tx] = libdevice.half2float(b_raw)
        cuda.syncthreads()
        for k in range(TILE):
            acc += As[ty, k] * Bs[k, tx]
        cuda.syncthreads()

    if row < M and col < N:
        res = alpha * acc
        if beta != float32(0.0):
            res += beta * libdevice.half2float(C[row * N + col])
        C[row * N + col] = libdevice.float2half_rn(res)


def solve(A, B, C, M, N, K, alpha, beta):
    """A, B, C are int16 views of float16 device arrays; alpha/beta are
    Python floats."""
    if M <= 0 or N <= 0 or K <= 0:
        return
    grid = ((N + TILE - 1) // TILE, (M + TILE - 1) // TILE)
    block = (TILE, TILE)
    gemm_kernel[grid, block](A, B, C, M, N, K,
                             float32(alpha), float32(beta))
    cuda.synchronize()


def _ref(a, b, c0, M, N, K, alpha, beta):
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    ref = alpha * (a64 @ b64) + beta * c0.astype(np.float64)
    return ref.astype(np.float16).astype(np.float64)


def _run(name, a, b, c0, M, N, K, alpha, beta):
    da = cuda.to_device(
        np.ascontiguousarray(a, dtype=np.float16).reshape(-1).view(np.int16))
    db = cuda.to_device(
        np.ascontiguousarray(b, dtype=np.float16).reshape(-1).view(np.int16))
    dc = cuda.to_device(
        np.ascontiguousarray(c0, dtype=np.float16).reshape(-1).view(np.int16))
    solve(da, db, dc, M, N, K, alpha, beta)
    got = dc.copy_to_host().view(np.float16).astype(np.float64).reshape(M, N)
    ref = _ref(a, b, c0, M, N, K, alpha, beta)
    worst = float(np.max(np.abs(got - ref)))
    mag = max(1.0, float(np.max(np.abs(ref))))
    ok = worst <= 0.02 + 0.01 * mag
    print(f"{name:>18} M={M} N={N} K={K}: worst_abs={worst:.3e} "
          f"{'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    def rnd(M, N):
        return rng.uniform(-2.0, 2.0, (M, N)).astype(np.float16)

    # Challenge example.
    ok &= _run("example", np.array([[1, 2, 3], [4, 5, 6]], np.float16),
               np.array([[1, 2], [3, 4], [5, 6]], np.float16),
               np.ones((2, 2), np.float16), 2, 2, 3, 1.0, 0.0)

    for M, N, K in ((16, 16, 16), (17, 19, 23), (32, 32, 32), (33, 65, 17),
                    (64, 64, 64), (100, 75, 50), (128, 128, 128),
                    (257, 131, 97)):
        ok &= _run("random", rnd(M, K), rnd(K, N), rnd(M, N), M, N, K, 1.0, 0.0)
        ok &= _run("random ab", rnd(M, K), rnd(K, N), rnd(M, N), M, N, K,
                   0.5, 0.5)

    # Benchmark: 1024 x 1024 x 1024.
    import statistics
    import time

    M = N = K = 1024
    a, b, c0 = rnd(M, K), rnd(K, N), rnd(M, N)
    da = cuda.to_device(a.reshape(-1).view(np.int16))
    db = cuda.to_device(b.reshape(-1).view(np.int16))
    dc = cuda.to_device(c0.reshape(-1).view(np.int16))
    solve(da, db, dc, M, N, K, 1.0, 0.0)  # warm-up, also triggers the JIT compile
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(da, db, dc, M, N, K, 1.0, 0.0)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    flops = 2.0 * M * N * K
    err = float(np.max(np.abs(
        dc.copy_to_host().view(np.float16).astype(np.float64).reshape(M, N)
        - _ref(a, b, c0, M, N, K, 1.0, 0.0))))
    print(f"Perf {M}x{N}x{K}: {ms:.4f} ms, {flops / (ms * 1e-3) / 1e12:.3f} "
          f"TFLOP/s, worst_abs={err:.2e}, {'OK' if err < 3 else 'FAIL'}")
    ok &= err < 3
    print("ALL OK" if ok else "FAILURES")
