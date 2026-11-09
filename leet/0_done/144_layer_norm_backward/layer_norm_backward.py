"""Numba CUDA port of layer_norm_backward.cu.

Layer-norm backward given the forward pass's mean/rstd:
dx = rstd*(g - mean(g) - xhat*mean(g*xhat)), dgamma = sum_i dy*xhat,
dbeta = sum_i dy.  Benchmark N=16,384, C=4,096 (x, dy, dx are 268 MB each).
The C++ kernel measures 10.7 ms (75 GB/s of the 804 MB minimum) against the
starter's 25.8 ms, whose parameter kernel is one thread per column summing all
rows serially in float32.

The port keeps the accuracy strategy without register blocking: R = 8 rows per
block accumulate dgamma/dbeta for the block's columns in two shared arrays
(each column is owned by one thread for the whole block, so the updates are
race-free), then one float64 atomic per column per parameter per block.  Row
data is re-read from L1 for the dx pass (Numba cannot hold 16 columns in
registers across a reduction without spilling).

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, float64, int32
from numba.cuda import libdevice

TPB = 256
R = 8
MAX_C_SHARED = 4096


@cuda.jit(cache=True)
def zero_kernel(a, n):
    i = int32(cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x)
    if i < n:
        a[i] = 0.0


@cuda.jit(cache=True)
def ln_bwd_kernel(x, gamma, mean, rstd, dy, dx, dgamma64, dbeta64, N, C):
    scratch = cuda.shared.array(8, dtype=float32)
    dgs = cuda.shared.array(MAX_C_SHARED, dtype=float32)
    dbs = cuda.shared.array(MAX_C_SHARED, dtype=float32)
    j = int32(cuda.threadIdx.x)
    nwarp = (int32(cuda.blockDim.x) + int32(31)) >> 5

    c = j
    while c < C:
        dgs[c] = float32(0.0)
        dbs[c] = float32(0.0)
        c += int32(TPB)
    cuda.syncthreads()

    row0 = int32(cuda.blockIdx.x) * int32(R)
    r = int32(0)
    while r < int32(R):
        row = row0 + r
        if row >= N:
            break
        mu = mean[row]
        rs = rstd[row]

        sg = float32(0.0)
        sgx = float32(0.0)
        c = j
        while c < C:
            xh = (x[row * C + c] - mu) * rs
            g = dy[row * C + c] * gamma[c]
            sg += g
            sgx += g * xh
            c += int32(TPB)
        sum_g = broadcast_reduce(sg, scratch, nwarp) / float32(C)
        sum_gx = broadcast_reduce(sgx, scratch, nwarp) / float32(C)

        c = j
        while c < C:
            xh = (x[row * C + c] - mu) * rs
            g = dy[row * C + c] * gamma[c]
            dx[row * C + c] = rs * (g - sum_g - xh * sum_gx)
            dgs[c] += dy[row * C + c] * xh
            dbs[c] += dy[row * C + c]
            c += int32(TPB)
        r += int32(1)

    cuda.syncthreads()
    c = j
    while c < C:
        if dgs[c] != float32(0.0):
            cuda.atomic.add(dgamma64, c, float64(dgs[c]))
        if dbs[c] != float32(0.0):
            cuda.atomic.add(dbeta64, c, float64(dbs[c]))
        c += int32(TPB)

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


@cuda.jit(cache=True)
def convert2_kernel(s0, s1, d0, d1, C):
    j = int32(cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x)
    if j < C:
        d0[j] = float32(s0[j])
        d1[j] = float32(s1[j])


_dg64 = None
_db64 = None


def solve(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C):
    """dgamma/dbeta are float32 device arrays; accumulation is float64."""
    global _dg64, _db64
    if N <= 0 or C <= 0:
        return
    if _dg64 is None or _dg64.size < C:
        _dg64 = cuda.device_array(C, dtype=np.float64)
        _db64 = cuda.device_array(C, dtype=np.float64)
    dg64 = _dg64[:C]
    db64 = _db64[:C]
    zero_kernel[(C + 255) // 256, 256](dg64, int32(C))
    zero_kernel[(C + 255) // 256, 256](db64, int32(C))
    blocks = (N + R - 1) // R
    ln_bwd_kernel[blocks, TPB](x, gamma, mean, rstd, dy, dx, dg64, db64,
                               int32(N), int32(C))
    convert2_kernel[(C + 255) // 256, 256](dg64, db64, dgamma, dbeta, int32(C))
    cuda.synchronize()

def _ref(x, gamma, mean, rstd, dy, N, C):
    x64 = x.reshape(N, C).astype(np.float64)
    dy64 = dy.reshape(N, C).astype(np.float64)
    g64 = gamma.astype(np.float64)
    mu = mean.astype(np.float64)[:, None]
    r = rstd.astype(np.float64)[:, None]
    xh = (x64 - mu) * r
    g = dy64 * g64
    dx = r * (g - g.mean(axis=1, keepdims=True) -
              xh * (g * xh).mean(axis=1, keepdims=True))
    dgamma = (dy64 * xh).sum(axis=0)
    dbeta = dy64.sum(axis=0)
    return dx.reshape(-1), dgamma, dbeta


def _check(name, x, gamma, mean, rstd, dy, N, C):
    rdx, rdg, rdb = _ref(x, gamma, mean, rstd, dy, N, C)
    dxp = cuda.to_device(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))
    dgp = cuda.to_device(np.ascontiguousarray(gamma, dtype=np.float32))
    dmp = cuda.to_device(np.ascontiguousarray(mean, dtype=np.float32))
    drp = cuda.to_device(np.ascontiguousarray(rstd, dtype=np.float32))
    dyp = cuda.to_device(np.ascontiguousarray(dy, dtype=np.float32).reshape(-1))
    dxo = cuda.device_array(N * C, dtype=np.float32)
    dgo = cuda.device_array(C, dtype=np.float32)
    dbo = cuda.device_array(C, dtype=np.float32)
    solve(dxp, dgp, dmp, drp, dyp, dxo, dgo, dbo, N, C)
    got_dx = dxo.copy_to_host().astype(np.float64)
    got_dg = dgo.copy_to_host().astype(np.float64)
    got_db = dbo.copy_to_host().astype(np.float64)
    e_dx = np.abs(got_dx - rdx)
    e_dg = np.abs(got_dg - rdg)
    e_db = np.abs(got_db - rdb)
    bad = int(np.count_nonzero(~(e_dx <= 1e-4 + 1e-4 * np.abs(rdx))))
    bad += int(np.count_nonzero(~(e_dg <= 1e-3 + 1e-4 * np.abs(rdg))))
    bad += int(np.count_nonzero(~(e_db <= 1e-3 + 1e-4 * np.abs(rdb))))
    print(f"{name:>16} N={N} C={C}: bad={bad} max_dx={e_dx.max():.2e} "
          f"max_dg={e_dg.max():.2e} max_db={e_db.max():.2e} "
          f"{'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    def make(N, C):
        x = rng.uniform(-250, 250, N * C).astype(np.float32)
        dy = rng.uniform(-10, 10, N * C).astype(np.float32)
        gamma = rng.uniform(-2, 2, C).astype(np.float32)
        x64 = x.reshape(N, C).astype(np.float64)
        mean = x64.mean(axis=1).astype(np.float32)
        var = ((x64 - mean.astype(np.float64)[:, None]) ** 2).mean(axis=1)
        rstd = (1.0 / np.sqrt(var + 1e-5)).astype(np.float32)
        return x, gamma, mean, rstd, dy

    for N, C in ((1, 1), (1, 7), (5, 3), (17, 33), (1000, 64), (3, 1024),
                 (2, 4096)):
        x, g, mu, r, dy = make(N, C)
        ok &= _check(f"{N}x{C}", x, g, mu, r, dy, N, C)

    # Benchmark: N=16384, C=4096.
    import statistics
    import time

    N, C = 16384, 4096
    x, g, mu, r, dy = make(N, C)
    ok &= _check("benchmark shape", x, g, mu, r, dy, N, C)
    dxp = cuda.to_device(x)
    dgp = cuda.to_device(g)
    dmp = cuda.to_device(mu)
    drp = cuda.to_device(r)
    dyp = cuda.to_device(dy)
    dxo = cuda.device_array(N * C, dtype=np.float32)
    dgo = cuda.device_array(C, dtype=np.float32)
    dbo = cuda.device_array(C, dtype=np.float32)
    solve(dxp, dgp, dmp, drp, dyp, dxo, dgo, dbo, N, C)  # warm-up / JIT
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        solve(dxp, dgp, dmp, drp, dyp, dxo, dgo, dbo, N, C)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    print(f"Perf N={N} C={C}: {ms:.4f} ms, "
          f"{3 * N * C * 4 / (ms * 1e6):.1f} GB/s")
    print("ALL OK" if ok else "FAILURES")
