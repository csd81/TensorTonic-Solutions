"""Numba CUDA port of int8_quantized_matmul.cu.

C = clamp(rint((sum_k (A-za)(B-zb) * sa*sb) / sc) + zc, -128, 127) for int8
matrices, int32 accumulation, round-half-to-even.  Benchmark: M=8192,
N=4096, K=2048 (6.87e10 int8 multiply-adds).

The C++ kernel (dp4a, 256x128 tile, 8x8 per thread) measures 68.99 ms
(996 G int8-MAC/s); the starter's 64x64/4x4 dp4a kernel measures 269 ms
after the alignment bug in its row-sum kernel is fixed (see the README).

Numba exposes no dp4a intrinsic, so this port computes the products with
scalar int8->int32 multiply-adds in a 64x64 tile with 4x4 per thread.  That
is four times the arithmetic instructions of the dp4a version before any
Numba overhead, so the port is expected to be several times slower; it stays
a faithful implementation of the same algorithm (exact int32 zero-point
correction with precomputed row/column sums and rintf in the epilogue).

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, int32, int8
from numba.cuda import libdevice

BM = 64
BN = 64
BK = 16
BKP = BK + 1  # shared-array strides must be literal global constants
BNP = BN + 1
TX = 16
TY = 16
TPB = TX * TY

_rs = None
_cs = None


@cuda.jit(device=True)
def qround(acc, rsum, csum, combined, zA, zB, zC, kz):
    exact = acc - zB * rsum - zA * csum + kz
    v = int32(libdevice.rintf(float32(exact) * combined)) + zC
    if v < -128:
        v = int32(-128)
    if v > 127:
        v = int32(127)
    return int8(v)


@cuda.jit(cache=True)
def row_sums_kernel(A, rs, M, K):
    r = int32(cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x)
    if r < M:
        s = int32(0)
        k = int32(0)
        while k < K:
            s += int32(A[r * K + k])
            k += int32(1)
        rs[r] = s


@cuda.jit(cache=True)
def col_sums_kernel(B, cs, K, N):
    c = int32(cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x)
    if c < N:
        s = int32(0)
        k = int32(0)
        while k < K:
            s += int32(B[k * N + c])
            k += int32(1)
        cs[c] = s


@cuda.jit(cache=True)
def qgemm_kernel(A, B, C, rs, cs, M, N, K, combined, zA, zB, zC):
    As = cuda.shared.array((BM, BKP), dtype=int8)
    Bs = cuda.shared.array((BK, BNP), dtype=int8)

    m0 = int32(cuda.blockIdx.y) * int32(BM)
    n0 = int32(cuda.blockIdx.x) * int32(BN)
    ty = int32(cuda.threadIdx.y)
    tx = int32(cuda.threadIdx.x)
    tid = ty * int32(TX) + tx

    a00 = int32(0)
    a01 = int32(0)
    a02 = int32(0)
    a03 = int32(0)
    a10 = int32(0)
    a11 = int32(0)
    a12 = int32(0)
    a13 = int32(0)
    a20 = int32(0)
    a21 = int32(0)
    a22 = int32(0)
    a23 = int32(0)
    a30 = int32(0)
    a31 = int32(0)
    a32 = int32(0)
    a33 = int32(0)

    t = int32(0)
    while t < K:
        idx = tid
        while idx < int32(BM * BK):
            row = idx // int32(BK)
            k = idx % int32(BK)
            gm = m0 + row
            gk = t + k
            if gm < M and gk < K:
                As[row, k] = A[gm * K + gk]
            else:
                As[row, k] = int8(0)
            idx += int32(TPB)
        idx = tid
        while idx < int32(BK * BN):
            kb = idx // int32(BN)
            col = idx % int32(BN)
            gk = t + kb
            gn = n0 + col
            if gk < K and gn < N:
                Bs[kb, col] = B[gk * N + gn]
            else:
                Bs[kb, col] = int8(0)
            idx += int32(TPB)
        cuda.syncthreads()

        for k in range(int32(BK)):
            r0 = int32(As[ty * int32(4), k])
            r1 = int32(As[ty * int32(4) + int32(1), k])
            r2 = int32(As[ty * int32(4) + int32(2), k])
            r3 = int32(As[ty * int32(4) + int32(3), k])
            c0 = int32(Bs[k, tx])
            c1 = int32(Bs[k, tx + int32(16)])
            c2 = int32(Bs[k, tx + int32(32)])
            c3 = int32(Bs[k, tx + int32(48)])
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

    kz = K * zA * zB
    gm = m0 + ty * int32(4)
    gn = n0 + tx
    if gm < M:
        if gn < N:
            C[gm * N + gn] = qround(a00, rs[gm], cs[gn], combined, zA, zB, zC, kz)
        if gn + int32(16) < N:
            C[gm * N + gn + int32(16)] = qround(a01, rs[gm], cs[gn + int32(16)], combined, zA, zB, zC, kz)
        if gn + int32(32) < N:
            C[gm * N + gn + int32(32)] = qround(a02, rs[gm], cs[gn + int32(32)], combined, zA, zB, zC, kz)
        if gn + int32(48) < N:
            C[gm * N + gn + int32(48)] = qround(a03, rs[gm], cs[gn + int32(48)], combined, zA, zB, zC, kz)
    if gm + int32(1) < M:
        if gn < N:
            C[(gm + int32(1)) * N + gn] = qround(a10, rs[gm + int32(1)], cs[gn], combined, zA, zB, zC, kz)
        if gn + int32(16) < N:
            C[(gm + int32(1)) * N + gn + int32(16)] = qround(a11, rs[gm + int32(1)], cs[gn + int32(16)], combined, zA, zB, zC, kz)
        if gn + int32(32) < N:
            C[(gm + int32(1)) * N + gn + int32(32)] = qround(a12, rs[gm + int32(1)], cs[gn + int32(32)], combined, zA, zB, zC, kz)
        if gn + int32(48) < N:
            C[(gm + int32(1)) * N + gn + int32(48)] = qround(a13, rs[gm + int32(1)], cs[gn + int32(48)], combined, zA, zB, zC, kz)
    if gm + int32(2) < M:
        if gn < N:
            C[(gm + int32(2)) * N + gn] = qround(a20, rs[gm + int32(2)], cs[gn], combined, zA, zB, zC, kz)
        if gn + int32(16) < N:
            C[(gm + int32(2)) * N + gn + int32(16)] = qround(a21, rs[gm + int32(2)], cs[gn + int32(16)], combined, zA, zB, zC, kz)
        if gn + int32(32) < N:
            C[(gm + int32(2)) * N + gn + int32(32)] = qround(a22, rs[gm + int32(2)], cs[gn + int32(32)], combined, zA, zB, zC, kz)
        if gn + int32(48) < N:
            C[(gm + int32(2)) * N + gn + int32(48)] = qround(a23, rs[gm + int32(2)], cs[gn + int32(48)], combined, zA, zB, zC, kz)
    if gm + int32(3) < M:
        if gn < N:
            C[(gm + int32(3)) * N + gn] = qround(a30, rs[gm + int32(3)], cs[gn], combined, zA, zB, zC, kz)
        if gn + int32(16) < N:
            C[(gm + int32(3)) * N + gn + int32(16)] = qround(a31, rs[gm + int32(3)], cs[gn + int32(16)], combined, zA, zB, zC, kz)
        if gn + int32(32) < N:
            C[(gm + int32(3)) * N + gn + int32(32)] = qround(a32, rs[gm + int32(3)], cs[gn + int32(32)], combined, zA, zB, zC, kz)
        if gn + int32(48) < N:
            C[(gm + int32(3)) * N + gn + int32(48)] = qround(a33, rs[gm + int32(3)], cs[gn + int32(48)], combined, zA, zB, zC, kz)


def solve(A, B, C, M, N, K, sA, sB, sC, zA, zB, zC):
    """A (M*K), B (K*N), C (M*N): int8 device arrays."""
    global _rs, _cs
    if M <= 0 or N <= 0 or K <= 0:
        return
    if _rs is None or _rs.size < M:
        _rs = cuda.device_array(M, dtype=np.int32)
    if _cs is None or _cs.size < N:
        _cs = cuda.device_array(N, dtype=np.int32)
    rs = _rs[:M]
    cs = _cs[:N]
    row_sums_kernel[(M + 255) // 256, 256](A, rs, int32(M), int32(K))
    col_sums_kernel[(N + 255) // 256, 256](B, cs, int32(K), int32(N))
    grid = ((N + BN - 1) // BN, (M + BM - 1) // BM)
    combined = np.float32((sA * sB) / sC)
    qgemm_kernel[grid, (TX, TY)](A, B, C, rs, cs, int32(M), int32(N), int32(K),
                                 combined, int32(zA), int32(zB), int32(zC))
    cuda.synchronize()


def _ref(a, b, M, N, K, combined, zA, zB, zC):
    a64 = a.reshape(M, K).astype(np.int64) - zA
    b64 = b.reshape(K, N).astype(np.int64) - zB
    dot = a64 @ b64
    scaled = dot.astype(np.float32) * combined
    v = np.rint(scaled).astype(np.int64) + zC
    return np.clip(v, -128, 127)


def _ref_sample(a, b, M, N, K, combined, zA, zB, zC, n=256):
    """Reference for a random sample of outputs (int64 dot per probe)."""
    rng = np.random.default_rng(3)
    rows = rng.integers(0, M, n)
    cols = rng.integers(0, N, n)
    a64 = a.reshape(M, K).astype(np.int64)
    b64 = b.reshape(K, N).astype(np.int64)
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        dot = int(np.dot(a64[rows[i]] - zA, b64[:, cols[i]] - zB))
        v = int(np.rint(np.float32(dot) * combined)) + zC
        out[i] = max(-128, min(127, v))
    return rows, cols, out


def _check(name, a, b, M, N, K, sA, sB, sC, zA, zB, zC, sampled=False):
    combined = np.float32((sA * sB) / sC)
    dA = cuda.to_device(np.ascontiguousarray(a, dtype=np.int8))
    dB = cuda.to_device(np.ascontiguousarray(b, dtype=np.int8))
    dC = cuda.device_array(M * N, dtype=np.int8)
    solve(dA, dB, dC, M, N, K, sA, sB, sC, zA, zB, zC)
    got = dC.copy_to_host().reshape(M, N).astype(np.int64)
    if sampled:
        rows, cols, ref = _ref_sample(a, b, M, N, K, combined, zA, zB, zC)
        bad = int(np.count_nonzero(got[rows, cols] != ref))
    else:
        ref = _ref(a, b, M, N, K, combined, zA, zB, zC)
        bad = int(np.count_nonzero(got != ref))
    print(f"{name:>20} M={M} N={N} K={K}: bad={bad} "
          f"{'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    ok &= _check("example 1", np.array([1, 2, 3, 4], np.int8),
                 np.array([5, 6, 7, 8], np.int8), 2, 2, 2, 0.1, 0.2, 0.05,
                 0, 0, 0)
    ok &= _check("example 2", np.array([1, 2], np.int8),
                 np.array([3, 4], np.int8), 1, 1, 2, 1.0, 1.0, 1.0, 1, 3, 5)
    for M, N, K, zA, zB, zC in ((1, 1, 1, 0, 0, 0), (5, 7, 11, 7, -3, 12),
                                (17, 33, 19, -128, 127, 127),
                                (64, 128, 64, 3, 5, 7),
                                (129, 130, 131, 0, 0, 0),
                                (256, 256, 257, 50, -50, 0)):
        a = rng.integers(-128, 128, M * K).astype(np.int8)
        b = rng.integers(-128, 128, K * N).astype(np.int8)
        s = rng.uniform(0.001, 0.1, 3)
        ok &= _check(f"{M}x{N}x{K}", a, b, M, N, K, s[0], s[1], s[2], zA, zB, zC)

    # Benchmark: M=8192, N=4096, K=2048.
    import statistics
    import time

    M, N, K = 8192, 4096, 2048
    a = rng.integers(-128, 128, M * K).astype(np.int8)
    b = rng.integers(-128, 128, K * N).astype(np.int8)
    ok &= _check("benchmark shape", a, b, M, N, K, 0.02, 0.03, 0.01, 5, -7, 3,
                 sampled=True)
    dA = cuda.to_device(a)
    dB = cuda.to_device(b)
    dC = cuda.device_array(M * N, dtype=np.int8)
    solve(dA, dB, dC, M, N, K, 0.02, 0.03, 0.01, 5, -7, 3)  # warm-up / JIT
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        solve(dA, dB, dC, M, N, K, 0.02, 0.03, 0.01, 5, -7, 3)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    print(f"Perf M={M} N={N} K={K}: {ms:.4f} ms, "
          f"{M * N * K / (ms * 1e-3) / 1e9:.1f} G int8-MAC/s")
    print("ALL OK" if ok else "FAILURES")
