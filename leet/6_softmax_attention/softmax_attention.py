"""Numba CUDA port of softmax_attention.cu.

Flash attention: out = softmax(Q K^T / sqrt(d)) V, one pass over N with the
online-softmax recurrence.

Mapping (same as the CUDA C++ version):
  * one warp owns ROWS_PER_WARP (= 2) rows of Q,
  * the 32 lanes of a warp are the 32 keys of the current K/V tile
    (BLOCK_N == warp size), so each lane computes one full score per row,
  * the running max / sum are warp-wide shuffle reductions,
  * K and V are staged through shared memory (padded by one float so the
    sK[lane][k] and sV[j][lane] accesses are bank-conflict free), and the
    next tile is prefetched into registers while the current one is used.

solve(Q, K, V, output, M, N, d) keeps the challenge signature.  Arrays may be
2-D or flat; host arrays are copied by Numba (device arrays are faster).
"""

import math

import numpy as np
from numba import cuda, float32

MAX_D = 128          # challenge constraint: 1 <= d <= 128
SHARED_D = 129       # padded row length (MAX_D + 1) for bank-conflict freedom
BLOCK_N = 32         # keys per tile == warp size
ROWS_PER_WARP = 2    # rows of Q handled by one warp
WARPS_PER_BLOCK = 8
BLOCK_M = ROWS_PER_WARP * WARPS_PER_BLOCK      # 16 rows per block
KV_ROWS_PER_WARP = BLOCK_N // WARPS_PER_BLOCK  # 4 K/V rows per warp
NEG_FLT_MAX = float32(-3.4028234663852886e38)


@cuda.jit(device=True)
def _warp_max(v):
    for off in (16, 8, 4, 2, 1):
        other = cuda.shfl_xor_sync(0xFFFFFFFF, v, off)
        if other > v:
            v = other
    return v


@cuda.jit(device=True)
def _warp_sum(v):
    for off in (16, 8, 4, 2, 1):
        v += cuda.shfl_xor_sync(0xFFFFFFFF, v, off)
    return v


@cuda.jit
def flash_attention_kernel(Q, K, V, output, M, N, d, scale):
    sQ = cuda.shared.array(shape=(BLOCK_M, SHARED_D), dtype=float32)
    sK = cuda.shared.array(shape=(BLOCK_N, SHARED_D), dtype=float32)
    sV = cuda.shared.array(shape=(BLOCK_N, SHARED_D), dtype=float32)
    sP = cuda.shared.array(shape=(BLOCK_M, BLOCK_N), dtype=float32)

    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5

    row0 = cuda.blockIdx.x * BLOCK_M + warp * ROWS_PER_WARP
    row1 = row0 + 1
    valid0 = row0 < M
    valid1 = row1 < M
    q0 = warp * ROWS_PER_WARP
    q1 = q0 + 1

    # Stage this warp's Q rows (zero-fill invalid rows).
    k = lane
    while k < d:
        if valid0:
            sQ[q0, k] = Q[row0 * d + k]
        else:
            sQ[q0, k] = 0.0
        if valid1:
            sQ[q1, k] = Q[row1 * d + k]
        else:
            sQ[q1, k] = 0.0
        k += 32

    m0 = NEG_FLT_MAX
    m1 = NEG_FLT_MAX
    l0 = float32(0.0)
    l1 = float32(0.0)
    # Each lane owns output elements lane, lane+32, lane+64, lane+96.
    o00 = float32(0.0)
    o01 = float32(0.0)
    o02 = float32(0.0)
    o03 = float32(0.0)
    o10 = float32(0.0)
    o11 = float32(0.0)
    o12 = float32(0.0)
    o13 = float32(0.0)

    # Software pipeline: tile t+1 is loaded into registers while tile t is
    # being used (the same prefetch structure as the CUDA C++ version).
    kbuf = cuda.local.array(shape=(4, 4), dtype=float32)
    vbuf = cuda.local.array(shape=(4, 4), dtype=float32)
    for r in range(KV_ROWS_PER_WARP):
        j = warp * KV_ROWS_PER_WARP + r
        gj = j  # first tile: k_start == 0
        for i in range(4):
            kk = lane + 32 * i
            if gj < N and kk < d:
                kbuf[r, i] = K[gj * d + kk]
                vbuf[r, i] = V[gj * d + kk]
            else:
                kbuf[r, i] = 0.0
                vbuf[r, i] = 0.0

    for k_start in range(0, N, BLOCK_N):
        # Publish the staged tile, then start loading the next one.
        for r in range(KV_ROWS_PER_WARP):
            j = warp * KV_ROWS_PER_WARP + r
            for i in range(4):
                kk = lane + 32 * i
                if kk < d:
                    sK[j, kk] = kbuf[r, i]
                    sV[j, kk] = vbuf[r, i]
        cuda.syncthreads()

        nxt = k_start + BLOCK_N
        for r in range(KV_ROWS_PER_WARP):
            j = warp * KV_ROWS_PER_WARP + r
            gj = nxt + j
            for i in range(4):
                kk = lane + 32 * i
                if nxt < N and gj < N and kk < d:
                    kbuf[r, i] = K[gj * d + kk]
                    vbuf[r, i] = V[gj * d + kk]
                else:
                    kbuf[r, i] = 0.0
                    vbuf[r, i] = 0.0

        # Scores: one key per lane, one dot product per row.
        dot0 = float32(0.0)
        dot1 = float32(0.0)
        for kk in range(d):
            kv = sK[lane, kk]
            dot0 += sQ[q0, kk] * kv
            dot1 += sQ[q1, kk] * kv
        s0 = dot0 * scale
        s1 = dot1 * scale
        if k_start + lane >= N:
            s0 = NEG_FLT_MAX
            s1 = NEG_FLT_MAX

        # Online softmax update.
        mb0 = _warp_max(s0)
        mb1 = _warp_max(s1)
        mn0 = m0
        if mb0 > mn0:
            mn0 = mb0
        mn1 = m1
        if mb1 > mn1:
            mn1 = mb1
        a0 = math.exp(m0 - mn0)
        a1 = math.exp(m1 - mn1)
        p0 = math.exp(s0 - mn0)
        p1 = math.exp(s1 - mn1)
        l0 = l0 * a0 + _warp_sum(p0)
        l1 = l1 * a1 + _warp_sum(p1)
        m0 = mn0
        m1 = mn1

        o00 *= a0
        o01 *= a0
        o02 *= a0
        o03 *= a0
        o10 *= a1
        o11 *= a1
        o12 *= a1
        o13 *= a1

        # Publish the tile's numerators and accumulate P*V for both rows.
        sP[q0, lane] = p0
        sP[q1, lane] = p1
        cuda.syncwarp()
        for j in range(BLOCK_N):
            pj0 = sP[q0, j]
            pj1 = sP[q1, j]
            idx = lane
            if idx < d:
                v = sV[j, idx]
                o00 += pj0 * v
                o10 += pj1 * v
            idx += 32
            if idx < d:
                v = sV[j, idx]
                o01 += pj0 * v
                o11 += pj1 * v
            idx += 32
            if idx < d:
                v = sV[j, idx]
                o02 += pj0 * v
                o12 += pj1 * v
            idx += 32
            if idx < d:
                v = sV[j, idx]
                o03 += pj0 * v
                o13 += pj1 * v
        cuda.syncthreads()

    inv0 = float32(1.0) / l0
    inv1 = float32(1.0) / l1
    idx = lane
    if idx < d:
        if valid0:
            output[row0 * d + idx] = o00 * inv0
        if valid1:
            output[row1 * d + idx] = o10 * inv1
    idx += 32
    if idx < d:
        if valid0:
            output[row0 * d + idx] = o01 * inv0
        if valid1:
            output[row1 * d + idx] = o11 * inv1
    idx += 32
    if idx < d:
        if valid0:
            output[row0 * d + idx] = o02 * inv0
        if valid1:
            output[row1 * d + idx] = o12 * inv1
    idx += 32
    if idx < d:
        if valid0:
            output[row0 * d + idx] = o03 * inv0
        if valid1:
            output[row1 * d + idx] = o13 * inv1


def _as_1d(array, rows, cols):
    if getattr(array, "ndim", 1) == 1:
        return array
    return array.reshape(rows * cols)


def solve(Q, K, V, output, M, N, d):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V, row-wise softmax."""
    if M < 1 or N < 1 or d < 1:
        raise ValueError("M, N and d must be positive")
    if d > MAX_D:
        raise ValueError(f"d must be <= {MAX_D}")

    Q1 = _as_1d(Q, M, d)
    K1 = _as_1d(K, N, d)
    V1 = _as_1d(V, N, d)
    O1 = _as_1d(output, M, d)

    scale = float32(1.0 / math.sqrt(d))
    blocks = (M + BLOCK_M - 1) // BLOCK_M
    flash_attention_kernel[blocks, WARPS_PER_BLOCK * 32](
        Q1, K1, V1, O1, M, N, d, scale)
    cuda.synchronize()


def _reference(Q, K, V, M, N, d):
    Qd = Q.astype(np.float64).reshape(M, d)
    Kd = K.astype(np.float64).reshape(N, d)
    Vd = V.astype(np.float64).reshape(N, d)
    scores = Qd @ Kd.T / math.sqrt(d)
    scores -= scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=1, keepdims=True)
    return p @ Vd


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    def run_case(name, Q, K, V, M, N, d):
        ref = _reference(Q, K, V, M, N, d)
        dQ = cuda.to_device(Q)
        dK = cuda.to_device(K)
        dV = cuda.to_device(V)
        dO = cuda.device_array(M * d, dtype=np.float32)
        solve(dQ, dK, dV, dO, M, N, d)
        got = dO.copy_to_host().reshape(M, d)

        max_err = float(np.max(np.abs(got - ref)))
        rel = max_err / max(1e-30, float(np.max(np.abs(ref))))
        print(f"{name:>18} M={M} N={N} d={d}: max_err={max_err:.3e} "
              f"rel={rel:.3e} {'OK' if rel < 1e-5 else 'FAIL'}")

    # Examples from the challenge.
    run_case("example 1",
             np.array([1, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32),
             np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.float32),
             np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                      dtype=np.float32), 2, 3, 4)
    run_case("example 2",
             np.array([1, 2], dtype=np.float32),
             np.array([1, 0, 0, 1], dtype=np.float32),
             np.array([3, 4, 5, 6], dtype=np.float32), 1, 2, 2)

    rng = np.random.default_rng(42)
    for M, N, d in [(1, 1, 1), (7, 5, 4), (33, 33, 3), (17, 33, 7),
                    (65, 100, 64), (100, 37, 128), (300, 1000, 128),
                    (64, 4096, 64), (2000, 64, 32)]:
        Q = rng.normal(0.0, 1.0, (M, d)).astype(np.float32)
        K = rng.normal(0.0, 1.0, (N, d)).astype(np.float32)
        V = rng.normal(0.0, 1.0, (N, d)).astype(np.float32)
        run_case("random", Q, K, V, M, N, d)

    # Host-array path (2-D inputs reshaped internally).
    M, N, d = 8, 12, 16
    Q = rng.normal(0.0, 1.0, (M, d)).astype(np.float32)
    K = rng.normal(0.0, 1.0, (N, d)).astype(np.float32)
    V = rng.normal(0.0, 1.0, (N, d)).astype(np.float32)
    out = np.zeros((M, d), dtype=np.float32)
    solve(Q, K, V, out, M, N, d)
    ref = _reference(Q, K, V, M, N, d)
    print(f"{'host arrays':>18}: max_err="
          f"{np.max(np.abs(out - ref)):.3e}")

    # Performance across a few shapes, median of 20 runs (device arrays).
    import statistics
    import time

    for M, N, d in [(512, 256, 128), (512, 256, 64), (2048, 1024, 128)]:
        Q = rng.normal(0.0, 1.0, (M, d)).astype(np.float32)
        K = rng.normal(0.0, 1.0, (N, d)).astype(np.float32)
        V = rng.normal(0.0, 1.0, (N, d)).astype(np.float32)
        dQ = cuda.to_device(Q)
        dK = cuda.to_device(K)
        dV = cuda.to_device(V)
        dO = cuda.device_array(M * d, dtype=np.float32)
        solve(dQ, dK, dV, dO, M, N, d)  # warm-up
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            solve(dQ, dK, dV, dO, M, N, d)
            times.append((time.perf_counter() - t0) * 1000.0)
        ms = statistics.median(times)
        got = dO.copy_to_host().reshape(M, d)
        ref = _reference(Q, K, V, M, N, d)
        rel = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
        print(f"Perf M={M} N={N} d={d}: {ms:.3f} ms, "
              f"{4.0 * M * N * d / (ms * 1e6):.1f} GFLOP/s, "
              f"rel_err={rel:.3e}, {'OK' if rel < 1e-5 else 'FAIL'}")
