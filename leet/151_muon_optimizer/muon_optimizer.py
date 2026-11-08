"""Numba CUDA port of muon_optimizer.cu.

Muon optimizer update: V = momentum * Buf + G, five quintic Newton-Schulz
steps on X0 = V / (||V||_F + 1e-7):
    A = X X^T,   B = b A + c A^2,   X <- a X + B X
and output = X * sqrt(max(M, N)).  When M > N the iteration runs on the
transposed matrix so the Gram matrix is K x K with K = min(M, N).

All matrix products use the same register-blocked GEMM as the CUDA C++
version (16x16 threads, 4x4 outputs per thread, 64x64 tiles) with a
residual epilogue C = alpha * A B + beta * R.  The two symmetric products
(X X^T and c A^2 + b A) compute only the upper-triangular tiles and a
mirror kernel fills the lower half.

solve(G, Buf, output, M, N, momentum) keeps the challenge signature.
"""

import math

import numpy as np
from numba import cuda, float32

BM = 64
BN = 64
BK = 16
BKP = 17
BNP = 65
TM = 4
TN = 4
THREADS = 256


@cuda.jit
def compute_v_kernel(G, Buf, X, norm_sq, M, N, momentum, transpose):
    sdata = cuda.shared.array(shape=THREADS, dtype=float32)
    tid = cuda.threadIdx.x
    total = M * N

    partial = float32(0.0)
    i = cuda.blockIdx.x * THREADS + tid
    stride = cuda.gridDim.x * THREADS
    while i < total:
        v = momentum * Buf[i] + G[i]
        partial += v * v
        if transpose:
            r = i // N
            c = i - r * N
            X[c * M + r] = v
        else:
            X[i] = v
        i += stride

    sdata[tid] = partial
    cuda.syncthreads()
    s = THREADS // 2
    while s > 0:
        if tid < s:
            sdata[tid] += sdata[tid + s]
        cuda.syncthreads()
        s //= 2
    if tid == 0:
        cuda.atomic.add(norm_sq, 0, sdata[0])


@cuda.jit
def normalize_kernel(X, norm_sq, total):
    inv = float32(1.0) / (math.sqrt(norm_sq[0]) + float32(1e-7))
    i = cuda.blockIdx.x * THREADS + cuda.threadIdx.x
    stride = cuda.gridDim.x * THREADS
    while i < total:
        X[i] *= inv
        i += stride


@cuda.jit
def gemm_kernel(A, B, C, M, N, K, lda, ldb, alpha, beta, R, ldr, trans_b,
                symmetric):
    As = cuda.shared.array(shape=(BM, BKP), dtype=float32)
    Bs = cuda.shared.array(shape=(BK, BNP), dtype=float32)

    if symmetric and cuda.blockIdx.x < cuda.blockIdx.y:
        return

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tid = ty * 16 + tx
    row0 = cuda.blockIdx.y * BM
    col0 = cuda.blockIdx.x * BN
    a_row = row0 + ty * TM
    b_col = col0 + tx * TN

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

    k0 = 0
    while k0 < K:
        for t in range(BM * BK // THREADS):
            idx = tid + t * THREADS
            r = idx // BK
            c = idx - r * BK
            if row0 + r < M and k0 + c < K:
                As[r, c] = A[(row0 + r) * lda + k0 + c]
            else:
                As[r, c] = 0.0
        for t in range(BK * BN // THREADS):
            idx = tid + t * THREADS
            if trans_b:
                r = idx % BK
                c = idx // BK
                if k0 + r < K and col0 + c < N:
                    Bs[r, c] = B[(col0 + c) * ldb + k0 + r]
                else:
                    Bs[r, c] = 0.0
            else:
                r = idx // BN
                c = idx - r * BN
                if k0 + r < K and col0 + c < N:
                    Bs[r, c] = B[(k0 + r) * ldb + col0 + c]
                else:
                    Bs[r, c] = 0.0
        cuda.syncthreads()

        for kk in range(BK):
            a0 = As[ty * TM + 0, kk]
            a1 = As[ty * TM + 1, kk]
            a2 = As[ty * TM + 2, kk]
            a3 = As[ty * TM + 3, kk]
            b0 = Bs[kk, tx * TN + 0]
            b1 = Bs[kk, tx * TN + 1]
            b2 = Bs[kk, tx * TN + 2]
            b3 = Bs[kk, tx * TN + 3]
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
        k0 += BK

    if a_row + 0 < M:
        if b_col + 0 < N:
            v00 = alpha * acc00
            if beta != 0.0:
                v00 += beta * R[(a_row + 0) * ldr + b_col + 0]
            C[(a_row + 0) * N + b_col + 0] = v00
        if b_col + 1 < N:
            v01 = alpha * acc01
            if beta != 0.0:
                v01 += beta * R[(a_row + 0) * ldr + b_col + 1]
            C[(a_row + 0) * N + b_col + 1] = v01
        if b_col + 2 < N:
            v02 = alpha * acc02
            if beta != 0.0:
                v02 += beta * R[(a_row + 0) * ldr + b_col + 2]
            C[(a_row + 0) * N + b_col + 2] = v02
        if b_col + 3 < N:
            v03 = alpha * acc03
            if beta != 0.0:
                v03 += beta * R[(a_row + 0) * ldr + b_col + 3]
            C[(a_row + 0) * N + b_col + 3] = v03
    if a_row + 1 < M:
        if b_col + 0 < N:
            v10 = alpha * acc10
            if beta != 0.0:
                v10 += beta * R[(a_row + 1) * ldr + b_col + 0]
            C[(a_row + 1) * N + b_col + 0] = v10
        if b_col + 1 < N:
            v11 = alpha * acc11
            if beta != 0.0:
                v11 += beta * R[(a_row + 1) * ldr + b_col + 1]
            C[(a_row + 1) * N + b_col + 1] = v11
        if b_col + 2 < N:
            v12 = alpha * acc12
            if beta != 0.0:
                v12 += beta * R[(a_row + 1) * ldr + b_col + 2]
            C[(a_row + 1) * N + b_col + 2] = v12
        if b_col + 3 < N:
            v13 = alpha * acc13
            if beta != 0.0:
                v13 += beta * R[(a_row + 1) * ldr + b_col + 3]
            C[(a_row + 1) * N + b_col + 3] = v13
    if a_row + 2 < M:
        if b_col + 0 < N:
            v20 = alpha * acc20
            if beta != 0.0:
                v20 += beta * R[(a_row + 2) * ldr + b_col + 0]
            C[(a_row + 2) * N + b_col + 0] = v20
        if b_col + 1 < N:
            v21 = alpha * acc21
            if beta != 0.0:
                v21 += beta * R[(a_row + 2) * ldr + b_col + 1]
            C[(a_row + 2) * N + b_col + 1] = v21
        if b_col + 2 < N:
            v22 = alpha * acc22
            if beta != 0.0:
                v22 += beta * R[(a_row + 2) * ldr + b_col + 2]
            C[(a_row + 2) * N + b_col + 2] = v22
        if b_col + 3 < N:
            v23 = alpha * acc23
            if beta != 0.0:
                v23 += beta * R[(a_row + 2) * ldr + b_col + 3]
            C[(a_row + 2) * N + b_col + 3] = v23
    if a_row + 3 < M:
        if b_col + 0 < N:
            v30 = alpha * acc30
            if beta != 0.0:
                v30 += beta * R[(a_row + 3) * ldr + b_col + 0]
            C[(a_row + 3) * N + b_col + 0] = v30
        if b_col + 1 < N:
            v31 = alpha * acc31
            if beta != 0.0:
                v31 += beta * R[(a_row + 3) * ldr + b_col + 1]
            C[(a_row + 3) * N + b_col + 1] = v31
        if b_col + 2 < N:
            v32 = alpha * acc32
            if beta != 0.0:
                v32 += beta * R[(a_row + 3) * ldr + b_col + 2]
            C[(a_row + 3) * N + b_col + 2] = v32
        if b_col + 3 < N:
            v33 = alpha * acc33
            if beta != 0.0:
                v33 += beta * R[(a_row + 3) * ldr + b_col + 3]
            C[(a_row + 3) * N + b_col + 3] = v33


@cuda.jit
def mirror_lower_kernel(C, K):
    tile = cuda.shared.array(shape=(32, 33), dtype=float32)
    bi = cuda.blockIdx.y
    bj = cuda.blockIdx.x
    if bi < bj:
        return
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    src_row = bj * 32 + ty
    src_col = bi * 32 + tx
    if src_row < K and src_col < K:
        tile[ty, tx] = C[src_row * K + src_col]
    cuda.syncthreads()
    dst_row = bi * 32 + ty
    dst_col = bj * 32 + tx
    if dst_row < K and dst_col < K:
        if bi == bj:
            if dst_row > dst_col:
                C[dst_row * K + dst_col] = tile[tx, ty]
        else:
            C[dst_row * K + dst_col] = tile[tx, ty]


@cuda.jit
def finalize_kernel(X, output, M, N, scale, transpose):
    total = M * N
    i = cuda.blockIdx.x * THREADS + cuda.threadIdx.x
    stride = cuda.gridDim.x * THREADS
    while i < total:
        if transpose:
            v = X[(i % N) * M + (i // N)]
        else:
            v = X[i]
        output[i] = v * scale
        i += stride


def _launch_gemm(A, B, C, M, N, K, lda, ldb, alpha, beta, R, ldr, trans_b,
                 symmetric=False):
    grid = ((N + BN - 1) // BN, (M + BM - 1) // BM)
    alpha = np.float32(alpha)
    beta = np.float32(beta)
    gemm_kernel[grid, (16, 16)](A, B, C, M, N, K, lda, ldb, alpha, beta, R,
                                ldr, trans_b, symmetric)


def solve(G, Buf, output, M, N, momentum):
    if G is None or Buf is None or output is None:
        return
    total = M * N
    if total <= 0:
        raise ValueError("M and N must be positive")

    momentum = np.float32(momentum)
    transpose = M > N
    K = N if transpose else M
    L = M if transpose else N

    a = float32(3.4445)
    b = float32(-4.7750)
    c = float32(2.0315)
    scale = float32(math.sqrt(L))

    kl = K * L
    kk = K * K
    workspace = cuda.device_array(2 * kl + 2 * kk + 1, dtype=np.float32)
    X_curr = workspace[:kl]
    X_next = workspace[kl:2 * kl]
    Amat = workspace[2 * kl:2 * kl + kk]
    Bmat = workspace[2 * kl + kk:2 * kl + 2 * kk]
    norm_sq = cuda.to_device(np.zeros(1, dtype=np.float32))

    elem_blocks = (total + THREADS - 1) // THREADS
    compute_v_kernel[elem_blocks, THREADS](G, Buf, X_curr, norm_sq, M, N,
                                           momentum, transpose)
    normalize_kernel[elem_blocks, THREADS](X_curr, norm_sq, total)

    ktile = (K + 31) // 32
    mirror_grid = (ktile, ktile)
    for _ in range(5):
        _launch_gemm(X_curr, X_curr, Amat, K, K, L, L, L, 1.0, 0.0, Amat,
                     0, True)
        mirror_lower_kernel[mirror_grid, (32, 32)](Amat, K)
        _launch_gemm(Amat, Amat, Bmat, K, K, K, K, K, c, b, Amat, K,
                     False, symmetric=True)
        mirror_lower_kernel[mirror_grid, (32, 32)](Bmat, K)
        _launch_gemm(Bmat, X_curr, X_next, K, L, K, K, L, 1.0, a, X_curr,
                     L, False)
        tmp = X_curr
        X_curr = X_next
        X_next = tmp

    finalize_kernel[elem_blocks, THREADS](X_curr, output, M, N, scale,
                                          transpose)
    cuda.synchronize()


def _reference(G, Buf, M, N, momentum):
    a, b, c = np.float32(3.4445), np.float32(-4.7750), np.float32(2.0315)
    transpose = M > N
    K = N if transpose else M
    L = M if transpose else N
    Gd = G.astype(np.float32).reshape(M, N)
    Bd = Buf.astype(np.float32).reshape(M, N)
    V = (momentum * Bd + Gd).astype(np.float32)
    if transpose:
        V = np.ascontiguousarray(V.T)
    V = (V / (np.float32(np.sqrt(np.sum(V * V, dtype=np.float32))) +
              np.float32(1e-7))).astype(np.float32)
    for _ in range(5):
        Amat = (V @ V.T).astype(np.float32)
        Bmat = (c * (Amat @ Amat) + b * Amat).astype(np.float32)
        V = (a * V + Bmat @ V).astype(np.float32)
    out = V * np.float32(np.sqrt(L))
    if transpose:
        out = np.ascontiguousarray(out.T)
    return out.reshape(-1)


def _run(name, G, Buf, M, N, momentum, tol=1e-3):
    ref = _reference(G, Buf, M, N, momentum)
    dG = cuda.to_device(G)
    dB = cuda.to_device(Buf)
    dO = cuda.device_array(M * N, dtype=np.float32)
    solve(dG, dB, dO, M, N, momentum)
    got = dO.copy_to_host()
    max_err = float(np.max(np.abs(got - ref)))
    max_ref = float(np.max(np.abs(ref)))
    rel = max_err / max(1e-30, max_ref)
    ok = rel < tol
    print(f"{name:>16} M={M} N={N} mom={momentum}: max_err={max_err:.3e} "
          f"rel={rel:.3e} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    ok = True
    rng = np.random.default_rng(42)

    # Challenge example (values from the statement).
    G = np.array([0.5, -1.0, 2.0, 1.0, 0.0, -0.5], dtype=np.float32)
    Buf = np.array([0.1, 0.2, 0.0, -0.3, 0.4, 0.1], dtype=np.float32)
    expected = np.array([0.5427371, -0.4092258, 1.1156158,
                         1.7146817, 0.5666907, -0.3777507], dtype=np.float32)
    dO = cuda.device_array(6, dtype=np.float32)
    solve(cuda.to_device(G), cuda.to_device(Buf), dO, 2, 3, np.float32(0.9))
    got = dO.copy_to_host()
    err = float(np.max(np.abs(got - expected)))
    print("example:", np.round(got, 7).tolist(),
          "OK" if err < 1e-5 else "FAIL")
    ok &= err < 1e-5

    for M, N, mom in [(1, 1, 0.9), (1, 5, 0.9), (5, 1, 0.9), (4, 4, 0.0),
                      (17, 33, 0.9), (33, 17, 0.9), (64, 64, 0.95),
                      (100, 37, 0.5), (37, 100, 0.5), (128, 256, 0.9),
                      (256, 256, 0.9), (512, 512, 0.9)]:
        n = M * N
        G = rng.normal(0.0, 1.0, n).astype(np.float32)
        Buf = rng.normal(0.0, 1.0, n).astype(np.float32)
        ok &= _run("random", G, Buf, M, N, mom)

    # Edge cases: all-zero and 1e-8 inputs.
    Z = np.zeros(32 * 32, dtype=np.float32)
    ok &= _run("all zero", Z, Z, 32, 32, 0.9)
    T = rng.uniform(-1e-8, 1e-8, 32 * 32).astype(np.float32)
    ok &= _run("1e-8 inputs", T, T, 32, 32, 0.9)

    # Performance: M = N = 1024, momentum = 0.9, median of 10 runs.
    import statistics
    import time

    M = N = 1024
    n = M * N
    G = rng.normal(0.0, 1.0, n).astype(np.float32)
    Buf = rng.normal(0.0, 1.0, n).astype(np.float32)
    dG = cuda.to_device(G)
    dB = cuda.to_device(Buf)
    dO = cuda.device_array(n, dtype=np.float32)
    solve(dG, dB, dO, M, N, 0.9)  # warm-up
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(dG, dB, dO, M, N, 0.9)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    gflops = 2.0 * 5.0 * (2.0 * 1024.0 ** 3 + 1024.0 ** 3) / (ms * 1e6)

    ref = _reference(G, Buf, M, N, 0.9)
    got = dO.copy_to_host()
    rel = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
    print(f"Perf 1024x1024: {ms:.3f} ms, {gflops:.1f} GFLOP/s, "
          f"rel={rel:.3e}, {'OK' if rel < 1e-3 else 'FAIL'}")
