"""Numba CUDA port of grouped_gemm.cu.

C[off[g]:off[g+1]] = A[off[g]:off[g+1]] . B[g]; benchmark G=8, M=8192,
K=1024, N=2048 (34.4 GFLOP).  The C++ kernel measures 46.0 ms (0.75 TFLOP/s);
the starter's 16x16 one-output-per-thread kernel measures 120 ms on the
benchmark shape but is only correct when group boundaries are 16-row aligned
(its shared B tile is filled from each thread's own group), so the corrected
naive baseline measures 394 ms.

This port keeps the C++ structure: a 1-thread planning kernel enumerates
(grupo, row-block) pairs into lookup arrays, and a register-blocked GEMM
(64x64 tile, 16x16 threads, 4x4 per thread, 16 scalar accumulators) processes
one group-aligned row block per grid.z slice, indexing A/C relative to the
block's first row and B by the group's weight matrix.  All index arithmetic
is int32.

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

_plans = None  # (blk_g, blk_row, blk_rows, nblk)


@cuda.jit(cache=True)
def plan_kernel(off, G, blk_g, blk_row, blk_rows, nblk, BM_):
    if cuda.blockIdx.x == 0 and cuda.threadIdx.x == 0:
        rb = int32(0)
        g = int32(0)
        while g < G:
            s = off[g]
            e = off[g + 1]
            rows = e - s
            r = int32(0)
            while r < rows:
                blk_g[rb] = g
                blk_row[rb] = s + r
                left = rows - r
                blk_rows[rb] = left if left < BM_ else BM_
                rb += int32(1)
                r += BM_
            g += int32(1)
        nblk[0] = rb


@cuda.jit(cache=True)
def grouped_gemm_kernel(A, B, C, blk_g, blk_row, blk_rows, nblk, N, K):
    As = cuda.shared.array((BM, BKP), dtype=float32)
    Bs = cuda.shared.array((BK, BNP), dtype=float32)

    rb = int32(cuda.blockIdx.z)
    if rb >= nblk[0]:
        return

    g = blk_g[rb]
    row0 = blk_row[rb]
    rows = blk_rows[rb]
    ab = row0 * K
    bb = g * K * N
    cb = row0 * N

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
    while t < K:
        idx = tid
        while idx < int32(BM * BK):
            row = idx // int32(BK)
            k = idx % int32(BK)
            gk = t + k
            if row < rows and gk < K:
                As[row, k] = A[ab + row * K + gk]
            else:
                As[row, k] = float32(0.0)
            idx += int32(TPB)
        idx = tid
        while idx < int32(BK * BN):
            kb = idx // int32(BN)
            col = idx % int32(BN)
            gk = t + kb
            gn = n0 + col
            if gk < K and gn < N:
                Bs[kb, col] = B[bb + gk * N + gn]
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

    gm = ty * int32(4)
    gn = n0 + tx
    if gm < rows:
        if gn < N:
            C[cb + gm * N + gn] = a00
        if gn + int32(16) < N:
            C[cb + gm * N + gn + int32(16)] = a01
        if gn + int32(32) < N:
            C[cb + gm * N + gn + int32(32)] = a02
        if gn + int32(48) < N:
            C[cb + gm * N + gn + int32(48)] = a03
    if gm + int32(1) < rows:
        if gn < N:
            C[cb + (gm + int32(1)) * N + gn] = a10
        if gn + int32(16) < N:
            C[cb + (gm + int32(1)) * N + gn + int32(16)] = a11
        if gn + int32(32) < N:
            C[cb + (gm + int32(1)) * N + gn + int32(32)] = a12
        if gn + int32(48) < N:
            C[cb + (gm + int32(1)) * N + gn + int32(48)] = a13
    if gm + int32(2) < rows:
        if gn < N:
            C[cb + (gm + int32(2)) * N + gn] = a20
        if gn + int32(16) < N:
            C[cb + (gm + int32(2)) * N + gn + int32(16)] = a21
        if gn + int32(32) < N:
            C[cb + (gm + int32(2)) * N + gn + int32(32)] = a22
        if gn + int32(48) < N:
            C[cb + (gm + int32(2)) * N + gn + int32(48)] = a23
    if gm + int32(3) < rows:
        if gn < N:
            C[cb + (gm + int32(3)) * N + gn] = a30
        if gn + int32(16) < N:
            C[cb + (gm + int32(3)) * N + gn + int32(16)] = a31
        if gn + int32(32) < N:
            C[cb + (gm + int32(3)) * N + gn + int32(32)] = a32
        if gn + int32(48) < N:
            C[cb + (gm + int32(3)) * N + gn + int32(48)] = a33


def solve(A, B, C, off, G, M, K, N):
    """A (M*K), B (G*K*N), C (M*N): float32 device arrays; off: int32 device
    array of length G+1."""
    global _plans
    if G <= 0 or M <= 0 or K <= 0 or N <= 0:
        return
    max_blocks = (M + BM - 1) // BM + G
    if _plans is None or _plans[0].size < max_blocks:
        _plans = (cuda.device_array(max_blocks, dtype=np.int32),
                  cuda.device_array(max_blocks, dtype=np.int32),
                  cuda.device_array(max_blocks, dtype=np.int32),
                  cuda.device_array(1, dtype=np.int32))
    blk_g, blk_row, blk_rows, nblk = _plans
    plan_kernel[1, 1](off, int32(G), blk_g, blk_row, blk_rows, nblk, int32(BM))
    grid = ((N + BN - 1) // BN, 1, max_blocks)
    grouped_gemm_kernel[grid, (TX, TY)](A, B, C, blk_g, blk_row, blk_rows, nblk,
                                        int32(N), int32(K))
    cuda.synchronize()


def _check(name, A, Bmat, off, G, M, K, N):
    a64 = A.reshape(M, K).astype(np.float64)
    b64 = Bmat.reshape(G, K, N).astype(np.float64)
    ref = np.empty((M, N), dtype=np.float64)
    for g in range(G):
        s, e = int(off[g]), int(off[g + 1])
        if e > s:
            ref[s:e] = a64[s:e] @ b64[g]
    dA = cuda.to_device(np.ascontiguousarray(A, dtype=np.float32))
    dB = cuda.to_device(np.ascontiguousarray(Bmat, dtype=np.float32))
    dOff = cuda.to_device(np.ascontiguousarray(off, dtype=np.int32))
    dC = cuda.device_array(M * N, dtype=np.float32)
    solve(dA, dB, dC, dOff, G, M, K, N)
    got = dC.copy_to_host().astype(np.float64).reshape(M, N)
    bad = 0
    max_abs = 0.0
    for g in range(G):
        s, e = int(off[g]), int(off[g + 1])
        if e <= s:
            continue
        err = np.abs(got[s:e] - ref[s:e])
        max_abs = max(max_abs, float(err.max()))
        bad += int(np.count_nonzero(~(err <= 1e-3 + 1e-4 * np.abs(ref[s:e]))))
    print(f"{name:>20} G={G} M={M} N={N} K={K}: bad={bad} "
          f"max_abs={max_abs:.2e} {'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    ok &= _check("example", np.array([1, 2, 3, 4, 5, 6, 7, 8], np.float32),
                 np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1], np.float32),
                 np.array([0, 2, 4], np.int32), 2, 4, 2, 3)
    ok &= _check("1x1x1", np.ones(1, np.float32), np.ones(1, np.float32),
                 np.array([0, 1], np.int32), 1, 1, 1, 1)
    ok &= _check("17 rows, empty grp",
                 rng.uniform(-1, 1, 17 * 19).astype(np.float32),
                 rng.uniform(-1, 1, 2 * 19 * 33).astype(np.float32),
                 np.array([0, 0, 17], np.int32), 2, 17, 19, 33)
    off = np.array([0, 3, 3, 10, 10], np.int32)
    ok &= _check("empty middle+tail",
                 rng.uniform(-1, 1, 10 * 13).astype(np.float32),
                 rng.uniform(-1, 1, 4 * 13 * 7).astype(np.float32), off, 4, 10,
                 13, 7)
    off = np.array([0, 1, 2, 100, 101, 300, 301, 302, 800], np.int32)
    ok &= _check("uneven groups",
                 rng.uniform(-1, 1, 800 * 48).astype(np.float32),
                 rng.uniform(-1, 1, 8 * 48 * 64).astype(np.float32), off, 8, 800,
                 48, 64)

    # Benchmark: G=8, M=8192, K=1024, N=2048 (equal groups).
    import statistics
    import time

    G, M, K, N = 8, 8192, 1024, 2048
    off = np.arange(0, M + 1, M // G, dtype=np.int32)
    A = rng.uniform(-1, 1, M * K).astype(np.float32)
    Bmat = rng.uniform(-1, 1, G * K * N).astype(np.float32)
    ok &= _check("benchmark shape", A, Bmat, off, G, M, K, N)
    dA = cuda.to_device(A)
    dB = cuda.to_device(Bmat)
    dOff = cuda.to_device(off)
    dC = cuda.device_array(M * N, dtype=np.float32)
    solve(dA, dB, dC, dOff, G, M, K, N)  # warm-up / JIT
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        solve(dA, dB, dC, dOff, G, M, K, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    print(f"Perf G={G} M={M} N={N} K={K}: {ms:.4f} ms, "
          f"{2.0 * M * K * N / (ms * 1e-3) / 1e12:.3f} TFLOP/s")
    print("ALL OK" if ok else "FAILURES")
