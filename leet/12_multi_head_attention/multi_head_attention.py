"""Numba CUDA port of multi_head_attention.cu.

Multi-head self-attention:
    head_i = softmax(Q_i K_i^T / sqrt(d_k)) V_i,  output = Concat(head_1..head_h)
where Q_i, K_i, V_i are the d_k-wide column slices of Q, K, V for head i
(d_k = d_model / h) and each row is an independent softmax.

Fast path (d_k <= 128) -- flash-attention style, same mapping as the CUDA C++
version:
  * one warp owns two rows of the head's output,
  * the 32 lanes of a warp are the 32 keys of the current K/V tile, so each
    lane computes one full score per row with a plain dot product,
  * the running max / sum are warp-wide shuffle reductions,
  * K/V tiles are staged in shared memory and the next tile is prefetched
    into local registers while the current one is used.

Slow path (d_k > 128) -- one block per (head, query row), block-wide dot
product reductions (same algorithm as the starter).

solve(Q, K, V, output, N, d_model, h) keeps the challenge signature.  Host
arrays are copied to the device and back by Numba.
"""

import math

import numpy as np
from numba import cuda, float32

MAX_DK = 128
SHARED_DK = 129
BLOCK_N = 32
ROWS_PER_WARP = 2
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
def mha_flash_kernel(Q, K, V, output, N, d_model, d_k, scale):
    sQ = cuda.shared.array(shape=(BLOCK_M, SHARED_DK), dtype=float32)
    sK = cuda.shared.array(shape=(BLOCK_N, SHARED_DK), dtype=float32)
    sV = cuda.shared.array(shape=(BLOCK_N, SHARED_DK), dtype=float32)
    sP = cuda.shared.array(shape=(BLOCK_M, BLOCK_N), dtype=float32)

    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    head_off = cuda.blockIdx.y * d_k

    row0 = cuda.blockIdx.x * BLOCK_M + warp * ROWS_PER_WARP
    row1 = row0 + 1
    valid0 = row0 < N
    valid1 = row1 < N
    q0 = warp * ROWS_PER_WARP
    q1 = q0 + 1

    # Stage this warp's Q rows (head slice, zero-fill invalid rows).
    k = lane
    while k < d_k:
        if valid0:
            sQ[q0, k] = Q[row0 * d_model + head_off + k]
        else:
            sQ[q0, k] = 0.0
        if valid1:
            sQ[q1, k] = Q[row1 * d_model + head_off + k]
        else:
            sQ[q1, k] = 0.0
        k += 32

    m0 = NEG_FLT_MAX
    m1 = NEG_FLT_MAX
    l0 = float32(0.0)
    l1 = float32(0.0)
    o00 = float32(0.0)
    o01 = float32(0.0)
    o02 = float32(0.0)
    o03 = float32(0.0)
    o10 = float32(0.0)
    o11 = float32(0.0)
    o12 = float32(0.0)
    o13 = float32(0.0)

    # Software pipeline: tile t+1 is loaded into registers while tile t is
    # being used.
    kbuf = cuda.local.array(shape=(4, 4), dtype=float32)
    vbuf = cuda.local.array(shape=(4, 4), dtype=float32)
    for r in range(KV_ROWS_PER_WARP):
        j = warp * KV_ROWS_PER_WARP + r
        gj = j  # first tile: k_start == 0
        for i in range(4):
            kk = lane + 32 * i
            if gj < N and kk < d_k:
                kbuf[r, i] = K[gj * d_model + head_off + kk]
                vbuf[r, i] = V[gj * d_model + head_off + kk]
            else:
                kbuf[r, i] = 0.0
                vbuf[r, i] = 0.0

    k_start = 0
    while k_start < N:
        # Publish the staged tile, then start loading the next one.
        for r in range(KV_ROWS_PER_WARP):
            j = warp * KV_ROWS_PER_WARP + r
            for i in range(4):
                kk = lane + 32 * i
                if kk < d_k:
                    sK[j, kk] = kbuf[r, i]
                    sV[j, kk] = vbuf[r, i]
        cuda.syncthreads()

        nxt = k_start + BLOCK_N
        for r in range(KV_ROWS_PER_WARP):
            j = warp * KV_ROWS_PER_WARP + r
            gj = nxt + j
            for i in range(4):
                kk = lane + 32 * i
                if nxt < N and gj < N and kk < d_k:
                    kbuf[r, i] = K[gj * d_model + head_off + kk]
                    vbuf[r, i] = V[gj * d_model + head_off + kk]
                else:
                    kbuf[r, i] = 0.0
                    vbuf[r, i] = 0.0

        # Scores: one key per lane, one dot product per row.
        dot0 = float32(0.0)
        dot1 = float32(0.0)
        for kk in range(d_k):
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
            if idx < d_k:
                v = sV[j, idx]
                o00 += pj0 * v
                o10 += pj1 * v
            idx += 32
            if idx < d_k:
                v = sV[j, idx]
                o01 += pj0 * v
                o11 += pj1 * v
            idx += 32
            if idx < d_k:
                v = sV[j, idx]
                o02 += pj0 * v
                o12 += pj1 * v
            idx += 32
            if idx < d_k:
                v = sV[j, idx]
                o03 += pj0 * v
                o13 += pj1 * v
        cuda.syncthreads()
        k_start += BLOCK_N

    inv0 = float32(1.0) / l0
    inv1 = float32(1.0) / l1
    idx = lane
    if idx < d_k:
        if valid0:
            output[row0 * d_model + head_off + idx] = o00 * inv0
        if valid1:
            output[row1 * d_model + head_off + idx] = o10 * inv1
    idx += 32
    if idx < d_k:
        if valid0:
            output[row0 * d_model + head_off + idx] = o01 * inv0
        if valid1:
            output[row1 * d_model + head_off + idx] = o11 * inv1
    idx += 32
    if idx < d_k:
        if valid0:
            output[row0 * d_model + head_off + idx] = o02 * inv0
        if valid1:
            output[row1 * d_model + head_off + idx] = o12 * inv1
    idx += 32
    if idx < d_k:
        if valid0:
            output[row0 * d_model + head_off + idx] = o03 * inv0
        if valid1:
            output[row1 * d_model + head_off + idx] = o13 * inv1


@cuda.jit
def mha_general_kernel(Q, K, V, output, N, d_model, d_k, scale):
    s_reduce = cuda.shared.array(shape=32, dtype=float32)
    s_scalar = cuda.shared.array(shape=2, dtype=float32)

    head = cuda.blockIdx.x
    q_row = cuda.blockIdx.y
    tid = cuda.threadIdx.x
    col = head * d_k

    q_val = float32(0.0)
    acc = float32(0.0)
    if tid < d_k:
        q_val = Q[q_row * d_model + col + tid]
    cuda.syncthreads()

    m_i = NEG_FLT_MAX
    l_i = float32(0.0)
    warp_id = tid >> 5
    lane_id = tid & 31
    num_warps = (cuda.blockDim.x + 31) >> 5

    j = 0
    while j < N:
        kv = j * d_model + col
        partial = float32(0.0)
        if tid < d_k:
            partial = q_val * K[kv + tid]
        for off in (16, 8, 4, 2, 1):
            partial += cuda.shfl_down_sync(0xFFFFFFFF, partial, off)
        if lane_id == 0:
            s_reduce[warp_id] = partial
        cuda.syncthreads()

        if warp_id == 0:
            total = float32(0.0)
            if lane_id < num_warps:
                total = s_reduce[lane_id]
            for off in (16, 8, 4, 2, 1):
                total += cuda.shfl_down_sync(0xFFFFFFFF, total, off)
            if lane_id == 0:
                score = total * scale
                m_new = m_i if m_i > score else score
                alpha = math.exp(m_i - m_new)
                beta = math.exp(score - m_new)
                l_i = l_i * alpha + beta
                m_i = m_new
                s_scalar[0] = alpha
                s_scalar[1] = beta
        cuda.syncthreads()

        if tid < d_k:
            acc = acc * s_scalar[0] + s_scalar[1] * V[kv + tid]
        cuda.syncthreads()
        j += 1

    if warp_id == 0 and lane_id == 0:
        s_scalar[0] = l_i
    cuda.syncthreads()
    if tid < d_k:
        output[q_row * d_model + col + tid] = acc / s_scalar[0]


def solve(Q, K, V, output, N, d_model, h):
    if Q is None or K is None or V is None or output is None:
        return
    if N <= 0 or d_model <= 0 or h <= 0:
        raise ValueError("N, d_model and h must be positive")
    if d_model % h != 0:
        raise ValueError("d_model must be divisible by h")

    d_k = d_model // h
    scale = float32(1.0 / math.sqrt(d_k))

    if d_k <= MAX_DK:
        grid = ((N + BLOCK_M - 1) // BLOCK_M, h)
        mha_flash_kernel[grid, WARPS_PER_BLOCK * 32](
            Q, K, V, output, N, d_model, d_k, scale)
    else:
        threads = ((d_k + 31) // 32) * 32
        mha_general_kernel[(h, N), threads](
            Q, K, V, output, N, d_model, d_k, scale)
    cuda.synchronize()


def _reference(Q, K, V, N, d_model, h):
    dk = d_model // h
    Qd = Q.astype(np.float64).reshape(N, d_model)
    Kd = K.astype(np.float64).reshape(N, d_model)
    Vd = V.astype(np.float64).reshape(N, d_model)
    out = np.zeros((N, d_model))
    for head in range(h):
        off = head * dk
        qh = Qd[:, off:off + dk]
        kh = Kd[:, off:off + dk]
        vh = Vd[:, off:off + dk]
        scores = qh @ kh.T / math.sqrt(dk)
        scores -= scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=1, keepdims=True)
        out[:, off:off + dk] = p @ vh
    return out


def _run(name, Q, K, V, N, d_model, h):
    ref = _reference(Q, K, V, N, d_model, h)
    dQ = cuda.to_device(Q)
    dK = cuda.to_device(K)
    dV = cuda.to_device(V)
    dO = cuda.device_array(N * d_model, dtype=np.float32)
    solve(dQ, dK, dV, dO, N, d_model, h)
    got = dO.copy_to_host().reshape(N, d_model)

    max_err = float(np.max(np.abs(got - ref)))
    rel = max_err / max(1e-30, float(np.max(np.abs(ref))))
    ok = rel < 1e-4
    print(f"{name:>16} N={N} d_model={d_model} h={h}: max_err={max_err:.3e} "
          f"rel={rel:.3e} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    ok = True
    rng = np.random.default_rng(42)

    # Challenge examples.
    ok &= _run("example 1",
               np.array([1, 0, 2, 3, 4, 5, 6, 7], dtype=np.float32),
               np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32),
               np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], dtype=np.float32),
               2, 4, 2)
    ok &= _run("example 2",
               np.array([1, 1], dtype=np.float32),
               np.array([1, 1], dtype=np.float32),
               np.array([2, 3], dtype=np.float32), 1, 2, 1)

    for N, d_model, h in [(1, 2, 1), (1, 8, 8), (5, 8, 4), (7, 16, 2),
                          (16, 64, 8), (33, 128, 4), (29, 96, 96),
                          (64, 256, 1), (100, 1024, 8), (64, 1024, 16),
                          (50, 512, 2)]:
        n = N * d_model
        Q = rng.normal(0.0, 1.0, n).astype(np.float32)
        K = rng.normal(0.0, 1.0, n).astype(np.float32)
        V = rng.normal(0.0, 1.0, n).astype(np.float32)
        ok &= _run("random", Q, K, V, N, d_model, h)

    # Performance at N=1024, d_model=1024 for h=8 and h=16 (fast path) and
    # h=4 (fallback path), median of 10 runs.
    import statistics
    import time

    for h in (8, 16, 4):
        N, d_model = 1024, 1024
        n = N * d_model
        Q = rng.uniform(-1.0, 1.0, n).astype(np.float32)
        K = rng.uniform(-1.0, 1.0, n).astype(np.float32)
        V = rng.uniform(-1.0, 1.0, n).astype(np.float32)
        dQ = cuda.to_device(Q)
        dK = cuda.to_device(K)
        dV = cuda.to_device(V)
        dO = cuda.device_array(n, dtype=np.float32)
        solve(dQ, dK, dV, dO, N, d_model, h)  # warm-up
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            solve(dQ, dK, dV, dO, N, d_model, h)
            times.append((time.perf_counter() - t0) * 1000.0)
        ms = statistics.median(times)
        gflops = 4.0 * N * N * d_model / (ms * 1e6)

        # Sampled correctness of the timed output.
        got = dO.copy_to_host().reshape(N, d_model)
        dk = d_model // h
        Qd = Q.astype(np.float64).reshape(N, d_model)
        Kd = K.astype(np.float64).reshape(N, d_model)
        Vd = V.astype(np.float64).reshape(N, d_model)
        max_err = 0.0
        max_ref = 0.0
        for head in range(0, h, max(1, h // 2)):
            off = head * dk
            for i in range(0, N, 97):
                s = Qd[i:i + 1, off:off + dk] @ Kd[:, off:off + dk].T
                s = s / math.sqrt(dk)
                s = np.exp(s - s.max())
                s /= s.sum()
                vals = s @ Vd[:, off:off + dk]
                for kk in range(0, dk, max(1, dk // 8)):
                    max_err = max(max_err, abs(float(got[i, off + kk]) -
                                               float(vals[0, kk])))
                    max_ref = max(max_ref, abs(float(vals[0, kk])))
        rel = max_err / max(1e-30, max_ref)
        print(f"Perf N={N} d_model={d_model} h={h}: {ms:.3f} ms, "
              f"{gflops:.1f} GFLOP/s, rel={rel:.3e}, "
              f"{'OK' if rel < 1e-4 else 'FAIL'}")
