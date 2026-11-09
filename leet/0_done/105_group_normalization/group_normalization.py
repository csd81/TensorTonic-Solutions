"""Numba CUDA port of group_normalization.cu.

For every (n, g): mean/var over the contiguous (C/G)*H*W group, then
y = gamma_c*(x-mu)*inv_std + beta_c.  Benchmark N=8, C=512, H=64, W=64, G=32
(256 groups of 65,536 elements; 201 MB of two-reads-plus-write traffic).

The C++ kernel measures 2.45 ms (82 GB/s, the wall) against the starter's
2.66 ms.  The port keeps the key simplification -- a group is one contiguous
range, so the base offset is computed once per block and the hot loops do no
index divisions -- with scalar loads (Numba has no float4), a warp-shuffle +
shared reduction of (sum, sumsq), a clamped variance and the subtract-form
affine transform.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, int32
from numba.cuda import libdevice

TPB = 256


@cuda.jit(cache=True)
def group_norm_kernel(X, gamma, beta, Y, C, HW, cpg, eps):
    s_sum = cuda.shared.array(8, dtype=float32)
    s_sq = cuda.shared.array(8, dtype=float32)

    n = int32(cuda.blockIdx.y)
    g = int32(cuda.blockIdx.x)
    c0 = g * cpg
    base = (n * C + c0) * HW
    count = cpg * HW

    s = float32(0.0)
    q = float32(0.0)
    i = int32(cuda.threadIdx.x)
    while i < count:
        v = X[base + i]
        s += v
        q += v * v
        i += int32(TPB)

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
    mu = s_sum[0] / float32(count)
    var = s_sq[0] / float32(count) - mu * mu
    if var < 0.0:
        var = float32(0.0)
    inv_std = libdevice.rsqrtf(var + eps)

    i = int32(cuda.threadIdx.x)
    while i < count:
        c = c0 + i // HW
        Y[base + i] = gamma[c] * ((X[base + i] - mu) * inv_std) + beta[c]
        i += int32(TPB)


def solve(X, gamma, beta, Y, N, C, H, W, G, eps):
    if N <= 0 or C <= 0 or G <= 0 or H <= 0 or W <= 0:
        return
    group_norm_kernel[(G, N), TPB](X, gamma, beta, Y, int32(C),
                                   int32(H * W), int32(C // G), float32(eps))
    cuda.synchronize()


def _ref(X, gamma, beta, N, C, H, W, G, eps):
    HW = H * W
    cpg = C // G
    x = X.reshape(N, C, HW).astype(np.float64)
    out = np.empty_like(x)
    for n in range(N):
        for g in range(G):
            blk = x[n, g * cpg:(g + 1) * cpg, :]
            mu = blk.mean()
            var = ((blk - mu) ** 2).mean()
            inv = 1.0 / np.sqrt(var + eps)
            gg = gamma[g * cpg:(g + 1) * cpg].astype(np.float64)[:, None]
            bb = beta[g * cpg:(g + 1) * cpg].astype(np.float64)[:, None]
            out[n, g * cpg:(g + 1) * cpg, :] = gg * (blk - mu) * inv + bb
    return out.reshape(N * C * HW)


def _check(name, X, gamma, beta, N, C, H, W, G, eps=1e-5):
    ref = _ref(X, gamma, beta, N, C, H, W, G, eps)
    dX = cuda.to_device(np.ascontiguousarray(X, dtype=np.float32).reshape(-1))
    dg = cuda.to_device(np.ascontiguousarray(gamma, dtype=np.float32))
    db = cuda.to_device(np.ascontiguousarray(beta, dtype=np.float32))
    dY = cuda.device_array(N * C * H * W, dtype=np.float32)
    solve(dX, dg, db, dY, N, C, H, W, G, eps)
    got = dY.copy_to_host().astype(np.float64)
    err = np.abs(got - ref)
    bad = int(np.count_nonzero(~(err <= 1e-4 + 1e-4 * np.abs(ref))))
    print(f"{name:>16} N={N} C={C} H={H} W={W} G={G}: bad={bad} "
          f"max_abs={err.max():.2e} {'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _check("example 1",
                 np.array([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 6, 6, 6, 6],
                          np.float32),
                 np.ones(4, np.float32), np.zeros(4, np.float32), 1, 4, 2, 2, 2)
    ok &= _check("example 2", np.array([1, 3, 2, 6], np.float32),
                 np.array([2, 1], np.float32), np.zeros(2, np.float32), 1, 2,
                 1, 2, 2)
    for N, C, H, W, G in ((1, 1, 1, 1, 1), (1, 4, 1, 1, 4), (2, 3, 5, 7, 1),
                          (2, 8, 4, 4, 2), (1, 64, 8, 8, 8),
                          (3, 16, 17, 13, 4), (1, 1024, 1, 1, 64)):
        X = rng.uniform(-100, 100, N * C * H * W).astype(np.float32)
        g = rng.uniform(0.1, 10, C).astype(np.float32)
        b = rng.uniform(-10, 10, C).astype(np.float32)
        ok &= _check(f"{N}x{C}x{H}x{W}x{G}", X, g, b, N, C, H, W, G)

    # Benchmark: N=8, C=512, H=64, W=64, G=32.
    import statistics
    import time

    N, C, H, W, G = 8, 512, 64, 64, 32
    X = rng.uniform(-100, 100, N * C * H * W).astype(np.float32)
    g = rng.uniform(0.1, 10, C).astype(np.float32)
    b = rng.uniform(-10, 10, C).astype(np.float32)
    ok &= _check("benchmark shape", X, g, b, N, C, H, W, G)
    dX = cuda.to_device(X)
    dg = cuda.to_device(g)
    db = cuda.to_device(b)
    dY = cuda.device_array(N * C * H * W, dtype=np.float32)
    solve(dX, dg, db, dY, N, C, H, W, G, 1e-5)  # warm-up / JIT
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(dX, dg, db, dY, N, C, H, W, G, 1e-5)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    print(f"Perf N={N} C={C} H={H} W={W} G={G}: {ms:.4f} ms, "
          f"{3 * N * C * H * W * 4 / (ms * 1e6):.1f} GB/s")
    print("ALL OK" if ok else "FAILURES")
