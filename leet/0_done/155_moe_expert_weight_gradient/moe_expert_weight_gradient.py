"""Numba CUDA port of moe_expert_weight_gradient.cu.

MoE expert weight gradients (grouped GEMM with a ragged reduction
dimension):
    dW[g] = X[s:e]^T * dY[s:e]   (K x N),  s = group_offsets[g],
                                          e = group_offsets[g + 1]
Experts are contiguous and sorted; empty groups must produce zeros.

The port uses the same tiling structure: one block per (64x64) output
tile of one expert, 256 threads (16x16), each computing a 4x4
register tile, with the reduction over the expert's token range staged
through 16-row shared tiles of X (transposed) and dY.

solve(X, dY, group_offsets, dW, G, M_total, K, N) keeps the challenge
signature.  Device arrays must be used for offsets and dW; host arrays
are copied by Numba.
"""

import numpy as np
from numba import cuda, float32

BM = 64
BN = 64
BK = 16
TM = 4
TN = 4
BMP = 65
BNP = 65
THREADS = 256


@cuda.jit
def moe_grad_kernel(X, dY, offsets, dW, G, K, N):
    As = cuda.shared.array(shape=(BK, BMP), dtype=float32)
    Bs = cuda.shared.array(shape=(BK, BNP), dtype=float32)

    g = cuda.blockIdx.z
    row_base = cuda.blockIdx.y * BM
    col_base = cuda.blockIdx.x * BN
    s = offsets[g]
    e = offsets[g + 1]

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tid = ty * 16 + tx

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

    m_tile = s
    while m_tile < e:
        for t in range(BK * BM // THREADS):
            idx = tid + t * THREADS
            m_off = idx // BM
            k_off = idx - m_off * BM
            m_idx = m_tile + m_off
            k_idx = row_base + k_off
            if m_idx < e and k_idx < K:
                As[m_off, k_off] = X[m_idx * K + k_idx]
            else:
                As[m_off, k_off] = 0.0
        for t in range(BK * BN // THREADS):
            idx = tid + t * THREADS
            m_off = idx // BN
            n_off = idx - m_off * BN
            m_idx = m_tile + m_off
            n_idx = col_base + n_off
            if m_idx < e and n_idx < N:
                Bs[m_off, n_off] = dY[m_idx * N + n_idx]
            else:
                Bs[m_off, n_off] = 0.0
        cuda.syncthreads()

        for m in range(BK):
            a0 = As[m, ty * TM + 0]
            a1 = As[m, ty * TM + 1]
            a2 = As[m, ty * TM + 2]
            a3 = As[m, ty * TM + 3]
            b0 = Bs[m, tx * TN + 0]
            b1 = Bs[m, tx * TN + 1]
            b2 = Bs[m, tx * TN + 2]
            b3 = Bs[m, tx * TN + 3]
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
        m_tile += BK

    # Every element of dW must be written; empty experts write zeros.
    out_row = row_base + ty * TM + 0
    if out_row < K:
        out_col = col_base + tx * TN + 0
        if out_col < N:
            dW[(g * K + out_row) * N + out_col] = acc00
        out_col = col_base + tx * TN + 1
        if out_col < N:
            dW[(g * K + out_row) * N + out_col] = acc01
        out_col = col_base + tx * TN + 2
        if out_col < N:
            dW[(g * K + out_row) * N + out_col] = acc02
        out_col = col_base + tx * TN + 3
        if out_col < N:
            dW[(g * K + out_row) * N + out_col] = acc03
    out_row = row_base + ty * TM + 1
    if out_row < K:
        out_col = col_base + tx * TN + 0
        if out_col < N:
            dW[(g * K + out_row) * N + out_col] = acc10
        out_col = col_base + tx * TN + 1
        if out_col < N:
            dW[(g * K + out_row) * N + out_col] = acc11
        out_col = col_base + tx * TN + 2
        if out_col < N:
            dW[(g * K + out_row) * N + out_col] = acc12
        out_col = col_base + tx * TN + 3
        if out_col < N:
            dW[(g * K + out_row) * N + out_col] = acc13
    out_row = row_base + ty * TM + 2
    if out_row < K:
        out_col = col_base + tx * TN + 0
        if out_col < N:
            dW[(g * K + out_row) * N + out_col] = acc20
        out_col = col_base + tx * TN + 1
        if out_col < N:
            dW[(g * K + out_row) * N + out_col] = acc21
        out_col = col_base + tx * TN + 2
        if out_col < N:
            dW[(g * K + out_row) * N + out_col] = acc22
        out_col = col_base + tx * TN + 3
        if out_col < N:
            dW[(g * K + out_row) * N + out_col] = acc23
    out_row = row_base + ty * TM + 3
    if out_row < K:
        out_col = col_base + tx * TN + 0
        if out_col < N:
            dW[(g * K + out_row) * N + out_col] = acc30
        out_col = col_base + tx * TN + 1
        if out_col < N:
            dW[(g * K + out_row) * N + out_col] = acc31
        out_col = col_base + tx * TN + 2
        if out_col < N:
            dW[(g * K + out_row) * N + out_col] = acc32
        out_col = col_base + tx * TN + 3
        if out_col < N:
            dW[(g * K + out_row) * N + out_col] = acc33


def solve(X, dY, group_offsets, dW, G, M_total, K, N):
    if X is None or dY is None or group_offsets is None or dW is None:
        return
    if G <= 0 or M_total <= 0 or K <= 0 or N <= 0:
        raise ValueError("dimensions must be positive")

    grid = ((N + BN - 1) // BN, (K + BM - 1) // BM, G)
    moe_grad_kernel[grid, (16, 16)](X, dY, group_offsets, dW, G, K, N)
    cuda.synchronize()


def _reference(X, dY, off, G, M_total, K, N):
    Xd = X.astype(np.float64).reshape(M_total, K)
    Yd = dY.astype(np.float64).reshape(M_total, N)
    out = np.zeros((G, K, N))
    for g in range(G):
        s, e = off[g], off[g + 1]
        if e > s:
            out[g] = Xd[s:e].T @ Yd[s:e]
    return out.reshape(-1)


def _run(name, X, dY, off, G, M_total, K, N, sampled=False):
    dX = cuda.to_device(X)
    dYv = cuda.to_device(dY)
    d_off = cuda.to_device(np.asarray(off, dtype=np.int32))
    dW = cuda.device_array(G * K * N, dtype=np.float32)
    solve(dX, dYv, d_off, dW, G, M_total, K, N)
    got = dW.copy_to_host()

    ref = _reference(X, dY, off, G, M_total, K, N)
    if sampled:
        idx = np.arange(0, G * K * N, max(1, G * K * N // 200000))
    else:
        idx = slice(None)
    max_err = float(np.max(np.abs(got[idx] - ref[idx])))
    max_ref = float(np.max(np.abs(ref[idx])))
    rel = max_err / max(1e-30, max_ref)
    ok = rel < 1e-4
    print(f"{name:>18} G={G} M={M_total} K={K} N={N}: "
          f"max_err={max_err:.3e} rel={rel:.3e} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    # Challenge example.
    ok &= _run("example", np.array([1, 2, 3, -1, 0, 4], dtype=np.float32),
               np.array([1, 0, 2, 1, -1, 3], dtype=np.float32),
               [0, 2, 2, 3], 3, 3, 2, 2)

    def random_case(G, M_total, K, N, two_empty, uniform, name):
        if uniform:
            counts = [M_total // G] * G
            counts[-1] += M_total % G
        else:
            w = rng.exponential(1.0, G)
            if two_empty and G > 2:
                w[0] = 0.0
                w[G // 2] = 0.0
            counts = np.floor(w / w.sum() * M_total).astype(np.int64)
            counts[-1] += M_total - counts.sum()
        off = [0]
        for c in counts:
            off.append(off[-1] + int(c))
        X = rng.uniform(-1.0, 1.0, M_total * K).astype(np.float32)
        dYv = rng.uniform(-1.0, 1.0, M_total * N).astype(np.float32)
        return _run(name, X, dYv, off, G, M_total, K, N)

    ok &= random_case(1, 1, 1, 1, False, True, "tiny")
    ok &= random_case(1, 100, 16, 16, False, True, "single expert")
    ok &= random_case(4, 33, 7, 5, True, False, "odd sizes")
    ok &= random_case(8, 1000, 64, 64, True, False, "skewed small")
    ok &= random_case(16, 4096, 128, 128, True, False, "skewed mid")
    ok &= random_case(5, 5000, 300, 70, True, False, "non-multiples")
    ok &= random_case(3, 300, 129, 65, True, True, "uniform split")
    ok &= random_case(64, 8192, 256, 128, True, False, "many experts")
    ok &= random_case(2, 17, 1024, 512, False, False, "long rows")

    # Performance: G=64, M_total=32768, K=1024, N=512, skewed, 2 empty.
    import statistics
    import time

    G, M_total, K, N = 64, 32768, 1024, 512
    w = rng.exponential(1.0, G)
    w[0] = 0.0
    w[G // 2] = 0.0
    counts = np.floor(w / w.sum() * M_total).astype(np.int64)
    counts[-1] += M_total - counts.sum()
    off = [0]
    for c in counts:
        off.append(off[-1] + int(c))
    X = rng.uniform(-1.0, 1.0, M_total * K).astype(np.float32)
    dYv = rng.uniform(-1.0, 1.0, M_total * N).astype(np.float32)
    dX = cuda.to_device(X)
    dYd = cuda.to_device(dYv)
    d_off = cuda.to_device(np.asarray(off, dtype=np.int32))
    dW = cuda.device_array(G * K * N, dtype=np.float32)
    solve(dX, dYd, d_off, dW, G, M_total, K, N)  # warm-up
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(dX, dYd, d_off, dW, G, M_total, K, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    gflops = 2.0 * M_total * K * N / (ms * 1e6)

    got = dW.copy_to_host().reshape(G, K, N)
    Xd = X.astype(np.float64).reshape(M_total, K)
    Yd = dYv.astype(np.float64).reshape(M_total, N)
    max_err = 0.0
    max_ref = 0.0
    for g in range(0, G, 8):
        s, e = off[g], off[g + 1]
        if e > s:
            ref = Xd[s:e].T @ Yd[s:e]
            max_err = max(max_err, float(np.max(np.abs(got[g] - ref))))
            max_ref = max(max_ref, float(np.max(np.abs(ref))))
    rel = max_err / max(1e-30, max_ref)
    print(f"Perf G={G} M={M_total} K={K} N={N}: {ms:.3f} ms, "
          f"{gflops:.1f} GFLOP/s, rel={rel:.3e}, "
          f"{'OK' if rel < 1e-4 else 'FAIL'}")
