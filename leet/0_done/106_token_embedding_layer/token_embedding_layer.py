"""Numba CUDA port of token_embedding_layer.cu.

For every (b, t): s = token_row + position_row, then layer-norm over D.
Benchmark B=32, T=512, V=30000, P=2048, D=768 (16,384 rows; two reads + one
write = 150 MB).  The C++ kernel measures 1.81 ms (83 GB/s, the memory wall)
against the starter's 2.17 ms.

Same structure: one block per (b, t), 256 threads, scalar loads over the row,
a warp-shuffle + shared reduction of (sum, sumsq), and a normalize pass that
re-reads the two rows (they are in L1 by then; Numba cannot hold the row in a
register across the reduction the way the C++ float4 path does).  The
variance is clamped at zero before rsqrtf and the affine transform is
evaluated in the challenge's subtract form.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, int32
from numba.cuda import libdevice

TPB = 256


@cuda.jit(cache=True)
def emb_ln_kernel(token_ids, position_ids, te, pe, gamma, beta, out, T, D, eps):
    s_sum = cuda.shared.array(8, dtype=float32)  # one slot per warp
    s_sq = cuda.shared.array(8, dtype=float32)

    b = int32(cuda.blockIdx.y)
    t = int32(cuda.blockIdx.x)
    ti = token_ids[b * T + t]
    pi = position_ids[t]
    tr = ti * D
    pr = pi * D
    orow = (b * T + t) * D

    s = float32(0.0)
    q = float32(0.0)
    d = int32(cuda.threadIdx.x)
    while d < D:
        v = te[tr + d] + pe[pr + d]
        s += v
        q += v * v
        d += int32(TPB)

    off = int32(16)
    while off > 0:
        s += cuda.shfl_down_sync(0xFFFFFFFF, s, off)
        q += cuda.shfl_down_sync(0xFFFFFFFF, q, off)
        off >>= 1
    lane = int32(cuda.threadIdx.x) & 31
    warp = int32(cuda.threadIdx.x) >> 5
    if lane == 0:
        s_sum[warp] = s
        s_sq[warp] = q
    cuda.syncthreads()
    if cuda.threadIdx.x == 0:
        ts = float32(0.0)
        tq = float32(0.0)
        w = int32(0)
        while w < int32(8):
            ts += s_sum[w]
            tq += s_sq[w]
            w += int32(1)
        s_sum[0] = ts
        s_sq[0] = tq
    cuda.syncthreads()
    mu = s_sum[0] / float32(D)
    var = s_sq[0] / float32(D) - mu * mu
    if var < 0.0:
        var = float32(0.0)
    inv_std = libdevice.rsqrtf(var + eps)

    d = int32(cuda.threadIdx.x)
    while d < D:
        v = te[tr + d] + pe[pr + d]
        out[orow + d] = gamma[d] * ((v - mu) * inv_std) + beta[d]
        d += int32(TPB)


def solve(token_ids, position_ids, te, pe, gamma, beta, out, B, T, V, P, D,
          eps):
    if B <= 0 or T <= 0 or D <= 0:
        return
    emb_ln_kernel[(T, B), TPB](token_ids, position_ids, te, pe, gamma, beta,
                               out, int32(T), int32(D), float32(eps))
    cuda.synchronize()


def _ref(tids, pids, te, pe, gamma, beta, B, T, V, P, D, eps):
    te64 = te.astype(np.float64)
    pe64 = pe.astype(np.float64)
    g = gamma.astype(np.float64)
    bta = beta.astype(np.float64)
    out = np.empty((B, T, D), dtype=np.float64)
    for b in range(B):
        for t in range(T):
            s = te64[tids[b, t]] + pe64[pids[t]]
            mu = s.mean()
            var = ((s - mu) ** 2).mean()
            out[b, t] = g * (s - mu) / np.sqrt(var + eps) + bta
    return out


def _check(name, tids, pids, te, pe, gamma, beta, B, T, V, P, D, eps=1e-5):
    ref = _ref(tids, pids, te, pe, gamma, beta, B, T, V, P, D, eps)
    dt = cuda.to_device(np.ascontiguousarray(tids, dtype=np.int32).reshape(-1))
    dp = cuda.to_device(np.ascontiguousarray(pids, dtype=np.int32))
    dte = cuda.to_device(np.ascontiguousarray(te, dtype=np.float32).reshape(-1))
    dpe = cuda.to_device(np.ascontiguousarray(pe, dtype=np.float32).reshape(-1))
    dg = cuda.to_device(np.ascontiguousarray(gamma, dtype=np.float32))
    db = cuda.to_device(np.ascontiguousarray(beta, dtype=np.float32))
    dout = cuda.device_array(B * T * D, dtype=np.float32)
    solve(dt, dp, dte, dpe, dg, db, dout, B, T, V, P, D, eps)
    got = dout.copy_to_host().astype(np.float64).reshape(B, T, D)
    err = np.abs(got - ref)
    bad = int(np.count_nonzero(~(err <= 1e-4 + 1e-4 * np.abs(ref))))
    print(f"{name:>18} B={B} T={T} D={D}: bad={bad} "
          f"max_abs={err.max():.2e} {'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    # Challenge example.
    ok &= _check("example", np.array([[2, 0]], np.int32),
                 np.array([0, 1], np.int32),
                 np.array([[1, 2, 3, 4], [0, 1, 0, -1], [2, 0, -2, 0]],
                          np.float32),
                 np.array([[0, 0, 0, 0], [1, 0, -1, 0]], np.float32),
                 np.ones(4, np.float32), np.zeros(4, np.float32), 1, 2, 3, 2,
                 4)
    for B, T, V, P, D in ((1, 1, 1, 1, 1), (1, 1, 3, 2, 4), (2, 3, 5, 4, 17),
                          (3, 7, 33, 5, 100), (1, 17, 64, 8, 129),
                          (4, 9, 1000, 16, 1024)):
        tids = rng.integers(0, V, (B, T)).astype(np.int32)
        pids = rng.integers(0, P, T).astype(np.int32)
        te = rng.uniform(-1, 1, (V, D)).astype(np.float32)
        pe = rng.uniform(-1, 1, (P, D)).astype(np.float32)
        gamma = rng.uniform(0.1, 10, D).astype(np.float32)
        beta = rng.uniform(-1, 1, D).astype(np.float32)
        ok &= _check(f"{B}x{T}x{D}", tids, pids, te, pe, gamma, beta, B, T, V,
                     P, D)

    # Benchmark: B=32, T=512, V=30000, P=2048, D=768.
    import statistics
    import time

    B, T, V, P, D = 32, 512, 30000, 2048, 768
    tids = rng.integers(0, V, (B, T)).astype(np.int32)
    pids = rng.integers(0, P, T).astype(np.int32)
    te = rng.uniform(-1, 1, (V, D)).astype(np.float32)
    pe = rng.uniform(-1, 1, (P, D)).astype(np.float32)
    gamma = rng.uniform(0.1, 10, D).astype(np.float32)
    beta = rng.uniform(-1, 1, D).astype(np.float32)
    ok &= _check("benchmark shape", tids, pids, te, pe, gamma, beta, B, T, V,
                 P, D)
    dt = cuda.to_device(np.ascontiguousarray(tids, dtype=np.int32).reshape(-1))
    dp = cuda.to_device(np.ascontiguousarray(pids, dtype=np.int32).reshape(-1))
    dte = cuda.to_device(np.ascontiguousarray(te, dtype=np.float32).reshape(-1))
    dpe = cuda.to_device(np.ascontiguousarray(pe, dtype=np.float32).reshape(-1))
    dg = cuda.to_device(gamma)
    db = cuda.to_device(beta)
    dout = cuda.device_array(B * T * D, dtype=np.float32)
    solve(dt, dp, dte, dpe, dg, db, dout, B, T, V, P, D, 1e-5)  # warm-up
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(dt, dp, dte, dpe, dg, db, dout, B, T, V, P, D, 1e-5)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    print(f"Perf B={B} T={T} D={D}: {ms:.4f} ms, "
          f"{3 * B * T * D * 4 / (ms * 1e6):.1f} GB/s")
    print("ALL OK" if ok else "FAILURES")
