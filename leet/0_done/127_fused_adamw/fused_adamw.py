"""Numba CUDA port of fused_adamw.cu.

One fused AdamW step: p *= (1-lr*wd); m = b1*m+(1-b1)*g;
v = b2*v+(1-b2)*g^2; p -= lr*(m/bc1)/(sqrt(v/bc2)+eps).

At the benchmark (N = 16,777,216) the kernel moves 448 MB (4 reads + 3
writes), so it is memory-bound: the C++ scalar and float4 kernels both
measure ~5.7 ms (82 GB/s) and the sqrt/division are fully hidden.  This port
keeps the scalar one-element-per-thread mapping (Numba has no float4 type).

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import math

import numpy as np
from numba import cuda, float32

THREADS = 256


@cuda.jit(cache=True)
def adamw_kernel(params, grad, m, v, N, lr, b1, b2, eps, wd, ibc1, ibc2):
    for i in range(cuda.grid(1), N, cuda.gridsize(1)):
        p = params[i]
        p *= float32(1.0) - lr * wd
        g = grad[i]
        mi = b1 * m[i] + (float32(1.0) - b1) * g
        vi = b2 * v[i] + (float32(1.0) - b2) * g * g
        mh = mi * ibc1
        vh = vi * ibc2
        p -= lr * mh / (math.sqrt(vh) + eps)
        params[i] = p
        m[i] = mi
        v[i] = vi


def solve(params, grad, m, v, N, lr, beta1, beta2, eps, weight_decay, t):
    """All four are float32 device arrays of length N; params, m, v are
    updated in place."""
    if N <= 0:
        return
    if params is None or grad is None or m is None or v is None:
        return
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    blocks = (N + THREADS - 1) // THREADS
    if blocks > 65535:
        blocks = 65535
    adamw_kernel[blocks, THREADS](
        params, grad, m, v, N,
        float32(lr), float32(beta1), float32(beta2), float32(eps),
        float32(weight_decay), float32(1.0 / bc1), float32(1.0 / bc2))
    cuda.synchronize()


def _ref(p0, g, m0, v0, lr, b1, b2, eps, wd, t):
    p = p0.astype(np.float64).copy()
    m = m0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    bc1 = 1.0 - b1 ** t
    bc2 = 1.0 - b2 ** t
    p *= 1.0 - lr * wd
    m[:] = b1 * m + (1.0 - b1) * g
    v[:] = b2 * v + (1.0 - b2) * g.astype(np.float64) ** 2
    p -= lr * (m / bc1) / (np.sqrt(v / bc2) + eps)
    return p, m, v


def _run(name, p0, g, m0, v0, lr, b1, b2, eps, wd, t):
    rp, rm, rv = _ref(p0, g, m0, v0, lr, b1, b2, eps, wd, t)
    dp = cuda.to_device(np.ascontiguousarray(p0, dtype=np.float32))
    dg = cuda.to_device(np.ascontiguousarray(g, dtype=np.float32))
    dm = cuda.to_device(np.ascontiguousarray(m0, dtype=np.float32))
    dv = cuda.to_device(np.ascontiguousarray(v0, dtype=np.float32))
    solve(dp, dg, dm, dv, p0.size, lr, b1, b2, eps, wd, t)
    gp = dp.copy_to_host().astype(np.float64)
    gm = dm.copy_to_host().astype(np.float64)
    gv = dv.copy_to_host().astype(np.float64)
    bad = 0
    for got, ref in ((gp, rp), (gm, rm), (gv, rv)):
        bad += int(np.count_nonzero(
            np.abs(got - ref) > 1e-5 + 1e-4 * np.abs(ref)))
    ok = bad == 0
    print(f"{name:>20} N={p0.size} t={t}: bad={bad} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([1.0, -1.0], np.float32),
               np.array([0.1, -0.1], np.float32), np.zeros(2, np.float32),
               np.zeros(2, np.float32), 0.1, 0.9, 0.999, 1e-8, 0.0, 1)
    ok &= _run("example 2", np.array([1.0], np.float32),
               np.array([0.5], np.float32), np.zeros(1, np.float32),
               np.zeros(1, np.float32), 0.1, 0.9, 0.999, 1e-8, 0.01, 1)
    for n, t, lr, wd, eps in ((1, 1, 1e-3, 0.0, 1e-8),
                              (1000, 1, 1e-3, 0.01, 1e-8),
                              (100000, 1000, 1e-2, 0.1, 1e-12),
                              (100000, 100000, 1.0, 1.0, 1e-2)):
        p = rng.uniform(-10.0, 10.0, n).astype(np.float32)
        g = rng.uniform(-10.0, 10.0, n).astype(np.float32)
        m = (rng.uniform(-10.0, 10.0, n) * 0.1).astype(np.float32)
        v = rng.uniform(0.0, 1.0, n).astype(np.float32)
        ok &= _run(f"random {n}", p, g, m, v, lr, 0.9, 0.999, eps, wd, t)

    # Benchmark: N = 16,777,216, t = 1000.
    import statistics
    import time

    N, t = 16777216, 1000
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
    p = rng.uniform(-10.0, 10.0, N).astype(np.float32)
    g = rng.uniform(-10.0, 10.0, N).astype(np.float32)
    m = (rng.uniform(-10.0, 10.0, N) * 0.1).astype(np.float32)
    v = rng.uniform(0.0, 1.0, N).astype(np.float32)
    dp = cuda.to_device(p)
    dg = cuda.to_device(g)
    dm = cuda.to_device(m)
    dv = cuda.to_device(v)
    solve(dp, dg, dm, dv, N, lr, b1, b2, eps, wd, t)  # warm-up + JIT compile
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(dp, dg, dm, dv, N, lr, b1, b2, eps, wd, t)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    bytes_moved = 7 * N * 4
    print(f"Perf N={N} t={t}: {ms:.4f} ms, "
          f"{bytes_moved / (ms * 1e6):.1f} GB/s")
    print("ALL OK" if ok else "FAILURES")
