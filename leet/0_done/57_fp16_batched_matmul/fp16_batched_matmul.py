"""Numba CUDA port of fp16_batched_matmul.cu.

C_b = A_b . B_b for BATCH batches of MxK @ KxN in FP16 with FP32
accumulation and half output.  Benchmark shape M=N=K=256 (the challenge does
not state the benchmark batch size; the C++ kernel measures 0.066 / 0.416 /
1.543 / 6.074 ms for B = 1/8/32/128, 2.2-2.8x faster than the starter).

Numba cannot index float16 arrays, so the buffers are passed as int16
bit-pattern views and converted with libdevice.half2float (tile load) and
libdevice.float2half_rn (epilogue).  Structure: 64x64 block tile, 16x16
threads, 4x4 per thread with 16 scalar accumulators, one grid.z slice per
batch, int32 index arithmetic.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, int32, int16
from numba.cuda import libdevice

BM = 64
BN = 64
BK = 32
BKP = BK + 1  # shared-array strides must be literal global constants
BNP = BN + 1
TX = 16
TY = 16
TPB = TX * TY


@cuda.jit(cache=True)
def bmm_fp16_kernel(A, B, C, M, N, K):
    As = cuda.shared.array((BM, BKP), dtype=float32)
    Bs = cuda.shared.array((BK, BNP), dtype=float32)

    b = int32(cuda.blockIdx.z)
    m0 = int32(cuda.blockIdx.y) * int32(BM)
    n0 = int32(cuda.blockIdx.x) * int32(BN)
    ty = int32(cuda.threadIdx.y)
    tx = int32(cuda.threadIdx.x)
    tid = ty * int32(TX) + tx
    # Batch slice offsets (int32: BATCH*M*K <= 128*1024*1024 < 2^31).
    ab = b * M * K
    bb = b * K * N
    cb = b * M * N

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
            gm = m0 + row
            gk = t + k
            if gm < M and gk < K:
                As[row, k] = libdevice.half2float(A[ab + gm * K + gk])
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
                Bs[kb, col] = libdevice.half2float(B[bb + gk * N + gn])
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
        if gn < N:
            C[cb + gm * N + gn] = libdevice.float2half_rn(a00)
        if gn + int32(16) < N:
            C[cb + gm * N + gn + int32(16)] = libdevice.float2half_rn(a01)
        if gn + int32(32) < N:
            C[cb + gm * N + gn + int32(32)] = libdevice.float2half_rn(a02)
        if gn + int32(48) < N:
            C[cb + gm * N + gn + int32(48)] = libdevice.float2half_rn(a03)
    if gm + int32(1) < M:
        gn = n0 + tx
        if gn < N:
            C[cb + (gm + int32(1)) * N + gn] = libdevice.float2half_rn(a10)
        if gn + int32(16) < N:
            C[cb + (gm + int32(1)) * N + gn + int32(16)] = libdevice.float2half_rn(a11)
        if gn + int32(32) < N:
            C[cb + (gm + int32(1)) * N + gn + int32(32)] = libdevice.float2half_rn(a12)
        if gn + int32(48) < N:
            C[cb + (gm + int32(1)) * N + gn + int32(48)] = libdevice.float2half_rn(a13)
    if gm + int32(2) < M:
        gn = n0 + tx
        if gn < N:
            C[cb + (gm + int32(2)) * N + gn] = libdevice.float2half_rn(a20)
        if gn + int32(16) < N:
            C[cb + (gm + int32(2)) * N + gn + int32(16)] = libdevice.float2half_rn(a21)
        if gn + int32(32) < N:
            C[cb + (gm + int32(2)) * N + gn + int32(32)] = libdevice.float2half_rn(a22)
        if gn + int32(48) < N:
            C[cb + (gm + int32(2)) * N + gn + int32(48)] = libdevice.float2half_rn(a23)
    if gm + int32(3) < M:
        gn = n0 + tx
        if gn < N:
            C[cb + (gm + int32(3)) * N + gn] = libdevice.float2half_rn(a30)
        if gn + int32(16) < N:
            C[cb + (gm + int32(3)) * N + gn + int32(16)] = libdevice.float2half_rn(a31)
        if gn + int32(32) < N:
            C[cb + (gm + int32(3)) * N + gn + int32(32)] = libdevice.float2half_rn(a32)
        if gn + int32(48) < N:
            C[cb + (gm + int32(3)) * N + gn + int32(48)] = libdevice.float2half_rn(a33)


def solve(A, B, C, BATCH, M, N, K):
    """int16 bit-pattern views of the half device arrays."""
    if BATCH <= 0 or M <= 0 or N <= 0 or K <= 0:
        return
    grid = ((N + BN - 1) // BN, (M + BM - 1) // BM, BATCH)
    bmm_fp16_kernel[grid, (TX, TY)](A, B, C, int32(M), int32(N), int32(K))
    cuda.synchronize()


def _check(name, a, bm, BATCH, M, N, K):
    ref = np.einsum("bmk,bkn->bmn",
                    a.reshape(BATCH, M, K).astype(np.float64),
                    bm.reshape(BATCH, K, N).astype(np.float64))
    dA = cuda.to_device(np.ascontiguousarray(a, dtype=np.float16).view(np.int16))
    dB = cuda.to_device(np.ascontiguousarray(bm, dtype=np.float16).view(np.int16))
    dC = cuda.device_array(BATCH * M * N, dtype=np.int16)
    solve(dA, dB, dC, BATCH, M, N, K)
    got = dC.copy_to_host().view(np.float16).astype(np.float64).reshape(BATCH, M, N)
    # The output is half: compare at half precision.
    err = np.abs(got - ref)
    bad = int(np.count_nonzero(~(err <= 1e-3 * np.abs(ref) + 1e-2)))
    print(f"{name:>16} B={BATCH} M={M} N={N} K={K}: bad={bad} "
          f"max_abs={err.max():.2e} {'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    ok &= _check("example",
                 np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], np.float16),
                 np.array([1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1], np.float16),
                 2, 2, 2, 3)
    for BATCH, M, N, K in ((1, 1, 1, 1), (3, 5, 7, 11), (7, 17, 33, 19),
                           (5, 64, 128, 64), (2, 129, 130, 131)):
        a = rng.uniform(-1, 1, BATCH * M * K).astype(np.float16)
        bm = rng.uniform(-1, 1, BATCH * K * N).astype(np.float16)
        ok &= _check(f"{BATCH}x{M}x{N}x{K}", a, bm, BATCH, M, N, K)

    # Benchmark: M=N=K=256, several batch sizes.
    import statistics
    import time

    M = N = K = 256
    for BATCH in (1, 8, 32, 128):
        a = rng.uniform(-1, 1, BATCH * M * K).astype(np.float16)
        bm = rng.uniform(-1, 1, BATCH * K * N).astype(np.float16)
        ok &= _check("benchmark", a, bm, BATCH, M, N, K)
        dA = cuda.to_device(a.view(np.int16))
        dB = cuda.to_device(bm.view(np.int16))
        dC = cuda.device_array(BATCH * M * N, dtype=np.int16)
        solve(dA, dB, dC, BATCH, M, N, K)  # warm-up / JIT
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            solve(dA, dB, dC, BATCH, M, N, K)
            times.append((time.perf_counter() - t0) * 1000.0)
        ms = statistics.median(times)
        print(f"Perf B={BATCH}: {ms:.4f} ms, "
              f"{2.0 * BATCH * M * N * K / (ms * 1e-3) / 1e12:.3f} TFLOP/s")
    print("ALL OK" if ok else "FAILURES")
