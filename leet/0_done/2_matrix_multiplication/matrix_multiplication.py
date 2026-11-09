"""Numba CUDA port of matrix_multiplication.cu.

C = A @ B, row-major float32:
  A: M x N, B: N x K, C: M x K.

The kernel uses the same register-blocked scheme as the CUDA version:
  * 16x16 threads per block (256 threads),
  * one block computes a 64x64 output tile,
  * each thread computes a 4x4 sub-tile with 16 register accumulators,
  * A (64x16) and B (16x64) tiles are staged in shared memory.

solve(A, B, C, M, N, K) keeps the challenge signature. A, B and C may be
Numba device arrays or ordinary NumPy arrays; Numba handles host copies.
Flat arrays are reshaped to 2-D before the kernel launch.
"""

import numpy as np
from numba import cuda, float32

BM = 64
BN = 64
BK = 16
TM = 4
TN = 4


@cuda.jit
def matrix_multiplication_kernel(A, B, C, M, N, K):
    s_A = cuda.shared.array((BM, BK), float32)
    s_B = cuda.shared.array((BK, BN), float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    tid = ty * cuda.blockDim.x + tx
    nthreads = cuda.blockDim.x * cuda.blockDim.y

    row0 = by * BM + ty * TM
    col0 = bx * BN + tx * TN

    acc00 = float32(0.0)
    acc01 = float32(0.0)
    acc02 = float32(0.0)
    acc03 = float32(0.0)
    acc10 = float32(0.0)
    acc11 = float32(0.0)
    acc12 = float32(0.0)
    acc13 = float32(0.0)
    acc20 = float32(0.0)
    acc21 = float32(0.0)
    acc22 = float32(0.0)
    acc23 = float32(0.0)
    acc30 = float32(0.0)
    acc31 = float32(0.0)
    acc32 = float32(0.0)
    acc33 = float32(0.0)

    for k0 in range(0, N, BK):
        # A tile: 64x16 (4 values per thread)
        for idx in range(tid, BM * BK, nthreads):
            r = idx // BK
            c = idx - r * BK
            gr = by * BM + r
            gc = k0 + c
            if gr < M and gc < N:
                s_A[r, c] = A[gr, gc]
            else:
                s_A[r, c] = float32(0.0)

        # B tile: 16x64 (4 values per thread)
        for idx in range(tid, BK * BN, nthreads):
            r = idx // BN
            c = idx - r * BN
            gr = k0 + r
            gc = bx * BN + c
            if gr < N and gc < K:
                s_B[r, c] = B[gr, gc]
            else:
                s_B[r, c] = float32(0.0)

        cuda.syncthreads()

        for kk in range(BK):
            a0 = s_A[ty * TM + 0, kk]
            a1 = s_A[ty * TM + 1, kk]
            a2 = s_A[ty * TM + 2, kk]
            a3 = s_A[ty * TM + 3, kk]

            b0 = s_B[kk, tx * TN + 0]
            b1 = s_B[kk, tx * TN + 1]
            b2 = s_B[kk, tx * TN + 2]
            b3 = s_B[kk, tx * TN + 3]

            acc00 += a0 * b0
            acc01 += a0 * b1
            acc02 += a0 * b2
            acc03 += a0 * b3
            acc10 += a1 * b0
            acc11 += a1 * b1
            acc12 += a1 * b2
            acc13 += a1 * b3
            acc20 += a2 * b0
            acc21 += a2 * b1
            acc22 += a2 * b2
            acc23 += a2 * b3
            acc30 += a3 * b0
            acc31 += a3 * b1
            acc32 += a3 * b2
            acc33 += a3 * b3

        cuda.syncthreads()

    if row0 + 0 < M and col0 + 0 < K:
        C[row0 + 0, col0 + 0] = acc00
    if row0 + 0 < M and col0 + 1 < K:
        C[row0 + 0, col0 + 1] = acc01
    if row0 + 0 < M and col0 + 2 < K:
        C[row0 + 0, col0 + 2] = acc02
    if row0 + 0 < M and col0 + 3 < K:
        C[row0 + 0, col0 + 3] = acc03
    if row0 + 1 < M and col0 + 0 < K:
        C[row0 + 1, col0 + 0] = acc10
    if row0 + 1 < M and col0 + 1 < K:
        C[row0 + 1, col0 + 1] = acc11
    if row0 + 1 < M and col0 + 2 < K:
        C[row0 + 1, col0 + 2] = acc12
    if row0 + 1 < M and col0 + 3 < K:
        C[row0 + 1, col0 + 3] = acc13
    if row0 + 2 < M and col0 + 0 < K:
        C[row0 + 2, col0 + 0] = acc20
    if row0 + 2 < M and col0 + 1 < K:
        C[row0 + 2, col0 + 1] = acc21
    if row0 + 2 < M and col0 + 2 < K:
        C[row0 + 2, col0 + 2] = acc22
    if row0 + 2 < M and col0 + 3 < K:
        C[row0 + 2, col0 + 3] = acc23
    if row0 + 3 < M and col0 + 0 < K:
        C[row0 + 3, col0 + 0] = acc30
    if row0 + 3 < M and col0 + 1 < K:
        C[row0 + 3, col0 + 1] = acc31
    if row0 + 3 < M and col0 + 2 < K:
        C[row0 + 3, col0 + 2] = acc32
    if row0 + 3 < M and col0 + 3 < K:
        C[row0 + 3, col0 + 3] = acc33


def _as_2d(array, rows, cols):
    """Accept both flat and 2-D arrays, including Numba device arrays."""
    if getattr(array, "ndim", 2) == 2:
        return array
    return array.reshape(rows, cols)


def solve(A, B, C, M, N, K):
    """Launch the kernel; C is written in place (device or host array)."""
    if M < 1 or N < 1 or K < 1:
        raise ValueError("M, N and K must be positive")

    A2 = _as_2d(A, M, N)
    B2 = _as_2d(B, N, K)
    C2 = _as_2d(C, M, K)

    threads = (16, 16)
    blocks = ((K + BN - 1) // BN, (M + BM - 1) // BM)
    matrix_multiplication_kernel[blocks, threads](A2, B2, C2, M, N, K)
    cuda.synchronize()


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(0)
    for M, N, K in ((2, 2, 2), (1, 3, 1), (17, 33, 19), (128, 257, 193)):
        A = rng.standard_normal((M, N)).astype(np.float32)
        B = rng.standard_normal((N, K)).astype(np.float32)
        C = np.zeros((M, K), dtype=np.float32)

        dA = cuda.to_device(A)
        dB = cuda.to_device(B)
        dC = cuda.to_device(C)
        solve(dA, dB, dC, M, N, K)
        got = dC.copy_to_host()

        ref = A.astype(np.float64) @ B.astype(np.float64)
        max_err = float(np.max(np.abs(got - ref)))
        rel = max_err / max(1.0, float(np.max(np.abs(ref))))
        print(f"M={M} N={N} K={K}: max_err={max_err:.3e} rel={rel:.3e} "
              f"{'OK' if rel < 1e-4 else 'FAIL'}")
