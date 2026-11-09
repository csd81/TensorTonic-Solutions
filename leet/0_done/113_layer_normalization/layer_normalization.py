"""Numba CUDA port of layer_normalization.cu.

y[i,j] = weight[j]*(x[i,j]-mu_i)/sqrt(var_i+eps) + bias[j] with per-row
statistics over C.  Benchmark N=65,536, C=512 (134 MB in + 134 MB out).

The C++ kernel (one block per row, one float4 per thread held in registers,
two block reductions over the registers) measures 3.21 ms at 84 GB/s against
the starter's 7.21 ms, which makes three scalar passes over each row.

The port keeps the register-resident structure: with C <= 1024 the chosen
block size covers the row in one element per thread, so the element stays in
a register across the two reductions and the write; larger C falls back to
strided loops that re-read the row (L1-resident).  Reductions are warp
shuffles + a shared combine, with the same barrier after the broadcast read
that the C++ needs before the second reduction can reuse the shared slot.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, int32
from numba.cuda import libdevice


@cuda.jit(cache=True)
def ln_kernel(x, w, b, y, N, C, eps):
    scratch = cuda.shared.array(32, dtype=float32)
    i = int32(cuda.blockIdx.x)
    nwarp = (int32(cuda.blockDim.x) + int32(31)) >> 5
    j = int32(cuda.threadIdx.x)
    single = int32(cuda.blockDim.x) >= C

    v = float32(0.0)
    if single and j < C:
        v = x[i * C + j]

    s = float32(0.0)
    if single:
        if j < C:
            s = v
    else:
        t = j
        while t < C:
            s += x[i * C + t]
            t += int32(cuda.blockDim.x)
    mu = broadcast_reduce(s, scratch, nwarp) / float32(C)

    q = float32(0.0)
    if single:
        if j < C:
            d = v - mu
            q = d * d
    else:
        t = j
        while t < C:
            d = x[i * C + t] - mu
            q += d * d
            t += int32(cuda.blockDim.x)
    var = broadcast_reduce(q, scratch, nwarp) / float32(C)
    inv_std = libdevice.rsqrtf(var + eps)

    if single:
        if j < C:
            y[i * C + j] = w[j] * ((v - mu) * inv_std) + b[j]
    else:
        t = j
        while t < C:
            y[i * C + t] = w[t] * ((x[i * C + t] - mu) * inv_std) + b[t]
            t += int32(cuda.blockDim.x)


@cuda.jit(device=True)
def broadcast_reduce(val, scratch, nwarp):
    v = val
    off = int32(16)
    while off > 0:
        v += cuda.shfl_down_sync(0xFFFFFFFF, v, off)
        off >>= 1
    lane = int32(cuda.threadIdx.x) & 31
    warp = int32(cuda.threadIdx.x) >> 5
    if lane == 0:
        scratch[warp] = v
    cuda.syncthreads()
    if cuda.threadIdx.x == 0:
        t = float32(0.0)
        k = int32(0)
        while k < nwarp:
            t += scratch[k]
            k += int32(1)
        scratch[0] = t
    cuda.syncthreads()
    r = scratch[0]
    cuda.syncthreads()  # the next reduction reuses scratch[0]
    return r


def _warp_round(n):
    t = ((n + 31) // 32) * 32
    return max(32, min(1024, t))


def solve(x, w, b, y, N, C, eps):
    if N <= 0 or C <= 0:
        return
    threads = _warp_round(C)
    ln_kernel[N, threads](x, w, b, y, int32(N), int32(C), float32(eps))
    cuda.synchronize()


def _ref(X, w, b, N, C, eps):
    x = X.reshape(N, C).astype(np.float64)
    mu = x.mean(axis=1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
    return (w.astype(np.float64) * (x - mu) / np.sqrt(var + eps) +
            b.astype(np.float64))


def _check(name, X, w, b, N, C, eps=1e-5):
    ref = _ref(X, w, b, N, C, eps)
    dx = cuda.to_device(np.ascontiguousarray(X, dtype=np.float32).reshape(-1))
    dw = cuda.to_device(np.ascontiguousarray(w, dtype=np.float32))
    db = cuda.to_device(np.ascontiguousarray(b, dtype=np.float32))
    dy = cuda.device_array(N * C, dtype=np.float32)
    solve(dx, dw, db, dy, N, C, eps)
    got = dy.copy_to_host().astype(np.float64).reshape(N, C)
    err = np.abs(got - ref)
    bad = int(np.count_nonzero(~(err <= 1e-4 + 1e-4 * np.abs(ref))))
    print(f"{name:>16} N={N} C={C}: bad={bad} "
          f"max_abs={err.max():.2e} {'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _check("example", np.array([1, 2, 3, 4, -1, 0, 0, 1], np.float32),
                 np.ones(4, np.float32), np.zeros(4, np.float32), 2, 4)
    for N, C in ((1, 1), (1, 7), (5, 3), (17, 33), (1000, 64), (3, 1024),
                 (2, 4096), (2, 1023)):
        X = rng.uniform(-100, 100, N * C).astype(np.float32)
        w = rng.uniform(0.1, 10, C).astype(np.float32)
        b = rng.uniform(-10, 10, C).astype(np.float32)
        ok &= _check(f"{N}x{C}", X, w, b, N, C)

    # Benchmark: N=65536, C=512.
    import statistics
    import time

    N, C = 65536, 512
    X = rng.uniform(-100, 100, N * C).astype(np.float32)
    w = rng.uniform(0.1, 10, C).astype(np.float32)
    b = rng.uniform(-10, 10, C).astype(np.float32)
    ok &= _check("benchmark shape", X, w, b, N, C)
    dx = cuda.to_device(X)
    dw = cuda.to_device(w)
    db = cuda.to_device(b)
    dy = cuda.device_array(N * C, dtype=np.float32)
    solve(dx, dw, db, dy, N, C, 1e-5)  # warm-up / JIT
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(dx, dw, db, dy, N, C, 1e-5)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    print(f"Perf N={N} C={C}: {ms:.4f} ms, {2 * N * C * 4 / (ms * 1e6):.1f} GB/s")
    print("ALL OK" if ok else "FAILURES")
