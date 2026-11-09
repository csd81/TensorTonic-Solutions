"""Numba CUDA port of sparse_matrix_dense_matrix_multiplication.cu.

C[M,K] = A[M,N] * B[N,K]; A is dense-stored but 60-70% zeros.  Benchmark:
M=4096, N=2048, K=512 (4.29e9 multiply-adds).  The C++ register-blocked GEMM
measures 6.8-7.3 ms (1.2 TFLOP/s dense-equivalent); the starter's 32x32 tile
measures 28.0 ms and this port measures 23.9 ms (0.36 TFLOP/s).

This port is the same register-blocked structure at a Numba-friendly size:
64x64 block tile, 16x16 threads, 4x4 per thread with 16 explicit scalar
accumulators (shared loads per multiply-add drop from 2 to 0.5).  All block
and index arithmetic is cast to int32: module-level Python ints are int64 in
Numba, and letting them into index expressions promotes every address
computation to 64-bit arithmetic, which costs more than it looks on sm_61.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, int32

BM = 64
BN = 64
BK = 32
BKP = BK + 1  # shared-array strides must be literal global constants
BNP = BN + 1
TX = 16
TY = 16
TPB = TX * TY


@cuda.jit(cache=True)
def spmm_kernel(A, B, C, M, N, K):
    As = cuda.shared.array((BM, BKP), dtype=float32)
    Bs = cuda.shared.array((BK, BNP), dtype=float32)

    m0 = int32(cuda.blockIdx.y) * int32(BM)
    n0 = int32(cuda.blockIdx.x) * int32(BN)
    ty = int32(cuda.threadIdx.y)
    tx = int32(cuda.threadIdx.x)
    tid = ty * int32(TX) + tx

    a00 = float32(0.0)
    a01 = float32(0.0)
    a02 = float32(0.0)
    a03 = float32(0.0)
    a10 = float32(0.0)
    a11 = float32(0.0)
    a12 = float32(0.0)
    a13 = float32(0.0)
    a20 = float32(0.0)
    a21 = float32(0.0)
    a22 = float32(0.0)
    a23 = float32(0.0)
    a30 = float32(0.0)
    a31 = float32(0.0)
    a32 = float32(0.0)
    a33 = float32(0.0)

    t = int32(0)
    while t < N:
        idx = tid
        while idx < int32(BM * BK):
            row = idx // int32(BK)
            k = idx % int32(BK)
            gm = m0 + row
            gk = t + k
            if gm < M and gk < N:
                As[row, k] = A[gm * N + gk]
            else:
                As[row, k] = float32(0.0)
            idx += int32(TPB)
        idx = tid
        while idx < int32(BK * BN):
            kb = idx // int32(BN)
            col = idx % int32(BN)
            gk = t + kb
            gn = n0 + col
            if gk < N and gn < K:
                Bs[kb, col] = B[gk * K + gn]
            else:
                Bs[kb, col] = float32(0.0)
            idx += int32(TPB)
        cuda.syncthreads()

        for k in range(int32(BK)):
            r0 = As[ty * int32(4), k]
            r1 = As[ty * int32(4) + int32(1), k]
            r2 = As[ty * int32(4) + int32(2), k]
            r3 = As[ty * int32(4) + int32(3), k]
            c0 = Bs[k, tx]
            c1 = Bs[k, tx + int32(16)]
            c2 = Bs[k, tx + int32(32)]
            c3 = Bs[k, tx + int32(48)]
            a00 += r0 * c0
            a01 += r0 * c1
            a02 += r0 * c2
            a03 += r0 * c3
            a10 += r1 * c0
            a11 += r1 * c1
            a12 += r1 * c2
            a13 += r1 * c3
            a20 += r2 * c0
            a21 += r2 * c1
            a22 += r2 * c2
            a23 += r2 * c3
            a30 += r3 * c0
            a31 += r3 * c1
            a32 += r3 * c2
            a33 += r3 * c3
        cuda.syncthreads()
        t += int32(BK)

    gm = m0 + ty * int32(4)
    if gm < M:
        gn = n0 + tx
        if gn < K:
            C[gm * K + gn] = a00
        if gn + int32(16) < K:
            C[gm * K + gn + int32(16)] = a01
        if gn + int32(32) < K:
            C[gm * K + gn + int32(32)] = a02
        if gn + int32(48) < K:
            C[gm * K + gn + int32(48)] = a03
    if gm + int32(1) < M:
        gn = n0 + tx
        if gn < K:
            C[(gm + int32(1)) * K + gn] = a10
        if gn + int32(16) < K:
            C[(gm + int32(1)) * K + gn + int32(16)] = a11
        if gn + int32(32) < K:
            C[(gm + int32(1)) * K + gn + int32(32)] = a12
        if gn + int32(48) < K:
            C[(gm + int32(1)) * K + gn + int32(48)] = a13
    if gm + int32(2) < M:
        gn = n0 + tx
        if gn < K:
            C[(gm + int32(2)) * K + gn] = a20
        if gn + int32(16) < K:
            C[(gm + int32(2)) * K + gn + int32(16)] = a21
        if gn + int32(32) < K:
            C[(gm + int32(2)) * K + gn + int32(32)] = a22
        if gn + int32(48) < K:
            C[(gm + int32(2)) * K + gn + int32(48)] = a23
    if gm + int32(3) < M:
        gn = n0 + tx
        if gn < K:
            C[(gm + int32(3)) * K + gn] = a30
        if gn + int32(16) < K:
            C[(gm + int32(3)) * K + gn + int32(16)] = a31
        if gn + int32(32) < K:
            C[(gm + int32(3)) * K + gn + int32(32)] = a32
        if gn + int32(48) < K:
            C[(gm + int32(3)) * K + gn + int32(48)] = a33


def solve(A, B, C, M, N, K, nnz):
    """A (M*N), B (N*K), C (M*K): float32 device arrays."""
    if M <= 0 or N <= 0 or K <= 0:
        return
    grid = ((K + BN - 1) // BN, (M + BM - 1) // BM)
    spmm_kernel[grid, (TX, TY)](A, B, C, int32(M), int32(N), int32(K))
    cuda.synchronize()


def _check(name, A, B, M, N, K, nnz):
    ref = A.reshape(M, N).astype(np.float64) @ B.reshape(N, K).astype(np.float64)
    dA = cuda.to_device(np.ascontiguousarray(A, dtype=np.float32))
    dB = cuda.to_device(np.ascontiguousarray(B, dtype=np.float32))
    dC = cuda.device_array(M * K, dtype=np.float32)
    solve(dA, dB, dC, M, N, K, nnz)
    got = dC.copy_to_host().astype(np.float64).reshape(M, K)
    err = np.abs(got - ref)
    # abs tolerance 1e-2: float32 accumulation of up to N=2048 terms of
    # magnitude ~100 leaves ~1.5e-3 absolute error at outputs whose true value
    # is near zero (cancellation).  numpy float32 shows the same spread.
    bad = int(np.count_nonzero(~(err <= 1e-2 + 1e-4 * np.abs(ref))))
    print(f"{name:>20} M={M} N={N} K={K}: bad={bad} "
          f"max_abs={err.max():.2e} {'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)

    def make(M, N, K):
        A = rng.uniform(-10, 10, M * N).astype(np.float32)
        A[rng.random(M * N) < 0.65] = 0.0
        B = rng.uniform(-10, 10, N * K).astype(np.float32)
        return A, B, int(np.count_nonzero(A))

    ok = True
    A = np.array([2, 0, 0, 1, 0, 3, 0, 0, 0, 0, 4, 0], np.float32)
    B = np.array([1, 2, 3, 4, 5, 6, 7, 8], np.float32)
    ok &= _check("example", A, B, 3, 4, 2, 3)
    for M, N, K in ((1, 1, 1), (5, 3, 7), (17, 33, 19), (64, 128, 64),
                    (100, 257, 129), (256, 512, 512), (129, 130, 131)):
        A, B, nnz = make(M, N, K)
        ok &= _check(f"{M}x{N}x{K}", A, B, M, N, K, nnz)

    # Benchmark: M=4096, N=2048, K=512.
    import statistics
    import time

    M, N, K = 4096, 2048, 512
    A, B, nnz = make(M, N, K)
    ok &= _check("benchmark shape", A, B, M, N, K, nnz)
    dA = cuda.to_device(A)
    dB = cuda.to_device(B)
    dC = cuda.device_array(M * K, dtype=np.float32)
    solve(dA, dB, dC, M, N, K, nnz)  # warm-up / JIT
    times = []
    for _ in range(8):
        t0 = time.perf_counter()
        solve(dA, dB, dC, M, N, K, nnz)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    print(f"Perf M={M} N={N} K={K}: {ms:.4f} ms, "
          f"{2 * M * N * K / (ms * 1e-3) / 1e12:.3f} TFLOP/s dense-equivalent")
    print("ALL OK" if ok else "FAILURES")
