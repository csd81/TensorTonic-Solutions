"""Numba CUDA port of matrix_power.cu.

output = input^P for a dense FP32 square matrix, N <= 1024, P <= 20;
benchmark N = 512.  The number of matrix products is fixed by the exponent
(P=20 needs 5), so the work is GEMM speed plus data movement.

The C++ kernel measures 2.1 ms at N=512, P=20 against the starter's 4.55 ms:
it uses a 64x128 register-blocked GEMM (4x8 per thread), left-to-right binary
exponentiation with the read-only input as the base, two ping-pong scratch
buffers (no per-product cudaMemcpy beyond the final one) and cached
workspace.

This port keeps that structure with a Numba-friendly 64x64 tile and 4x4 per
thread (16 explicit scalar accumulators).  Block and index arithmetic is
cast to int32 because module-level Python ints are int64 in Numba and
otherwise promote every address computation to 64-bit.  The ping-pong
buffers are Numba device arrays swapped by reference in Python.

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

_w0 = None
_w1 = None


@cuda.jit(cache=True)
def gemm_kernel(A, B, C, N):
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
            if gm < N and gk < N:
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
            if gk < N and gn < N:
                Bs[kb, col] = B[gk * N + gn]
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
    if gm < N:
        gn = n0 + tx
        if gn < N:
            C[gm * N + gn] = a00
        if gn + int32(16) < N:
            C[gm * N + gn + int32(16)] = a01
        if gn + int32(32) < N:
            C[gm * N + gn + int32(32)] = a02
        if gn + int32(48) < N:
            C[gm * N + gn + int32(48)] = a03
    if gm + int32(1) < N:
        gn = n0 + tx
        if gn < N:
            C[(gm + int32(1)) * N + gn] = a10
        if gn + int32(16) < N:
            C[(gm + int32(1)) * N + gn + int32(16)] = a11
        if gn + int32(32) < N:
            C[(gm + int32(1)) * N + gn + int32(32)] = a12
        if gn + int32(48) < N:
            C[(gm + int32(1)) * N + gn + int32(48)] = a13
    if gm + int32(2) < N:
        gn = n0 + tx
        if gn < N:
            C[(gm + int32(2)) * N + gn] = a20
        if gn + int32(16) < N:
            C[(gm + int32(2)) * N + gn + int32(16)] = a21
        if gn + int32(32) < N:
            C[(gm + int32(2)) * N + gn + int32(32)] = a22
        if gn + int32(48) < N:
            C[(gm + int32(2)) * N + gn + int32(48)] = a23
    if gm + int32(3) < N:
        gn = n0 + tx
        if gn < N:
            C[(gm + int32(3)) * N + gn] = a30
        if gn + int32(16) < N:
            C[(gm + int32(3)) * N + gn + int32(16)] = a31
        if gn + int32(32) < N:
            C[(gm + int32(3)) * N + gn + int32(32)] = a32
        if gn + int32(48) < N:
            C[(gm + int32(3)) * N + gn + int32(48)] = a33


def solve(inp, out, N, P):
    """inp and out are flat N*N float32 device arrays; inp is never modified."""
    global _w0, _w1
    if N <= 0 or P <= 0:
        return
    if P == 1:
        out.copy_to_device(inp)
        cuda.synchronize()
        return
    if _w0 is None or _w0.size < N * N:
        _w0 = cuda.device_array(N * N, dtype=np.float32)
        _w1 = cuda.device_array(N * N, dtype=np.float32)
    grid = ((N + BN - 1) // BN, (N + BM - 1) // BM)
    # Slice exact-size views: the cached buffers are never shrunk, and
    # copy_to_device requires matching shapes.
    R = _w0[:N * N]
    T = _w1[:N * N]
    R.copy_to_device(inp)
    msb = P.bit_length() - 1
    for bit in range(msb - 1, -1, -1):
        gemm_kernel[grid, (TX, TY)](R, R, T, int32(N))
        R, T = T, R
        if (P >> bit) & 1:
            gemm_kernel[grid, (TX, TY)](R, inp, T, int32(N))
            R, T = T, R
    out.copy_to_device(R)
    cuda.synchronize()


def _ref(a, N, P):
    x = a.astype(np.float64).reshape(N, N)
    result = None
    base = x.copy()
    e = P
    while e > 0:
        if e & 1:
            result = base.copy() if result is None else result @ base
        e >>= 1
        if e > 0:
            base = base @ base
    return result


def _check(name, a, N, P):
    ref = _ref(a, N, P)
    max_ref = np.abs(ref).max() if ref.size else 0.0
    din = cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    dout = cuda.device_array(N * N, dtype=np.float32)
    solve(din, dout, N, P)
    got = dout.copy_to_host().astype(np.float64).reshape(N, N)
    err = np.abs(got - ref)
    tol = 1e-4 * np.minimum(max_ref, 1e30) + 1e-4 * np.abs(ref)
    bad = int(np.count_nonzero(~(err <= tol)))
    print(f"{name:>18} N={N} P={P}: bad={bad} max_err={err.max():.2e} "
          f"max_ref={max_ref:.2e} {'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)

    def make(N, scale):
        return (scale * rng.uniform(-1, 1, N * N)).astype(np.float32)

    ok = True
    ok &= _check("example 1", np.array([1, 2, 3, 4], np.float32), 2, 3)
    ok &= _check("example 2",
                 np.array([1, 0, 2, 0, 1, 0, 3, 0, 0], np.float32), 3, 2)
    ok &= _check("P=1 copy", np.array([1, 2, 3, 4], np.float32), 2, 1)
    for N, P in ((1, 1), (1, 20), (2, 2), (8, 3), (16, 5), (33, 7),
                 (64, 10), (100, 20), (129, 11), (256, 8)):
        ok &= _check(f"{N}x{N} P={P}", make(N, 0.9 / np.sqrt(N)), N, P)
    ok &= _check("unscaled 64 P=3", make(64, 10.0), 64, 3)

    # Benchmark: N=512.
    import statistics
    import time

    N = 512
    a = make(N, 0.9 / np.sqrt(N))
    ok &= _check("benchmark P=2", a, N, 2)
    ok &= _check("benchmark P=20", a, N, 20)
    din = cuda.to_device(a)
    dout = cuda.device_array(N * N, dtype=np.float32)
    for P in (2, 20):
        solve(din, dout, N, P)  # warm-up / JIT
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            solve(din, dout, N, P)
            times.append((time.perf_counter() - t0) * 1000.0)
        ms = statistics.median(times)
        print(f"Perf N={N} P={P}: {ms:.4f} ms")
    print("ALL OK" if ok else "FAILURES")
