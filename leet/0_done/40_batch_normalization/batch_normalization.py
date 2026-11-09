"""Numba CUDA port of batch_normalization.cu.

y[i,j] = gamma[j]*(x[i,j]-mu[j])/sqrt(var[j]+eps) + beta[j] for [N, C]
row-major tensors.  Benchmark N=5000; the C++ kernel measures 0.757 ms at
C=1024 (81 GB/s, the memory wall for two reads + one write) against the
starter's 9.16 ms (column-strided reads, one block per channel).

The port keeps the coalesced three-kernel structure: a stats kernel with
consecutive threads owning consecutive channels (FP32 per-thread partials,
per-channel atomic accumulation), a finalize kernel folding gamma and
inv_std into a per-channel scale and keeping mu separate (the write pass uses
scale*(x-mu)+beta, which avoids the catastrophic cancellation of the folded
x*scale+shift form when the output is small), and a scalar write pass (Numba
has no float4).  The per-channel accumulators are float32 here; that is
accurate for the benchmark data and the variance is clamped at zero before
rsqrtf (the one-pass estimate can round negative when the spread is tiny).

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, float64, int32
from numba.cuda import libdevice

THREADS = 256
ROWS_PER_BLOCK = 64

_sums = None
_sumsq = None
_scale = None
_mu = None


@cuda.jit(cache=True)
def zero_kernel(a):
    i = int32(cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x)
    if i < a.size:
        a[i] = 0.0


@cuda.jit(cache=True)
def stats_kernel(x, sums, sumsq, N, C):
    row0 = int32(cuda.blockIdx.x) * int32(ROWS_PER_BLOCK)
    row1 = row0 + int32(ROWS_PER_BLOCK)
    if row1 > N:
        row1 = N
    if row0 >= N:
        return
    bd = int32(cuda.blockDim.x)
    j = int32(cuda.threadIdx.x)
    while j < C:
        # Double accumulation: the one-pass variance cancels, and with tiny N
        # a float32 partial leaves the variance dominated by rounding.
        # float64 atomics are supported on sm_60+.
        s = float64(0.0)
        q = float64(0.0)
        i = row0
        while i < row1:
            v = float64(x[i * C + j])
            s += v
            q += v * v
            i += int32(1)
        cuda.atomic.add(sums, j, s)
        cuda.atomic.add(sumsq, j, q)
        j += bd


@cuda.jit(cache=True)
def finalize_kernel(sums, sumsq, gamma, scale, mu, N, C, eps):
    j = int32(cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x)
    if j < C:
        # Variance from the one-pass sums is a difference of two ~mean^2
        # terms; do it in float64 (only C threads) so a small spread is not
        # swamped by float32 cancellation -- the same choice as the C++.
        m = float64(sums[j]) / float64(N)
        var = float64(sumsq[j]) / float64(N) - m * m
        if var < 0.0:
            var = float64(0.0)
        inv_std = libdevice.rsqrtf(float32(var) + eps)
        scale[j] = gamma[j] * inv_std
        mu[j] = float32(m)


@cuda.jit(cache=True)
def apply_kernel(x, y, scale, mu, beta, N, C):
    total = N * C
    stride = int32(cuda.blockDim.x * cuda.gridDim.x)
    i = int32(cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x)
    while i < total:
        j = i % C
        y[i] = (x[i] - mu[j]) * scale[j] + beta[j]
        i += stride


def solve(x, gamma, beta, y, N, C, eps):
    """All tensors are float32 device arrays (gamma/beta of length C)."""
    global _sums, _sumsq, _scale, _mu
    if N <= 0 or C <= 0:
        return
    if _sums is None or _sums.size < C:
        _sums = cuda.device_array(C, dtype=np.float64)
        _sumsq = cuda.device_array(C, dtype=np.float64)
        _scale = cuda.device_array(C, dtype=np.float32)
        _mu = cuda.device_array(C, dtype=np.float32)
    zero_kernel[(C + 255) // 256, 256](_sums)
    zero_kernel[(C + 255) // 256, 256](_sumsq)
    stats_blocks = (N + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    threads = 256 if C >= 256 else ((C + 31) // 32) * 32
    stats_kernel[stats_blocks, threads](x, _sums, _sumsq, int32(N), int32(C))
    finalize_kernel[(C + 255) // 256, 256](_sums, _sumsq, gamma, _scale, _mu,
                                           int32(N), int32(C), float32(eps))
    total = N * C
    blocks = min(2048, max(1, (total + 255) // 256))
    apply_kernel[blocks, 256](x, y, _scale, _mu, beta, int32(N), int32(C))
    cuda.synchronize()


def _ref(x, gamma, beta, N, C, eps):
    x64 = x.reshape(N, C).astype(np.float64)
    mu = x64.mean(axis=0)
    var = ((x64 - mu) ** 2).mean(axis=0)
    return gamma.astype(np.float64) * (x64 - mu) / np.sqrt(var + eps) + \
        beta.astype(np.float64)


def _check(name, x, gamma, beta, N, C, eps=1e-5):
    ref = _ref(x, gamma, beta, N, C, eps)
    dx = cuda.to_device(np.ascontiguousarray(x, dtype=np.float32))
    dg = cuda.to_device(np.ascontiguousarray(gamma, dtype=np.float32))
    db = cuda.to_device(np.ascontiguousarray(beta, dtype=np.float32))
    dy = cuda.device_array(N * C, dtype=np.float32)
    solve(dx, dg, db, dy, N, C, eps)
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
    ok &= _check("example 1", np.array([1, 2, 3, 4, 5, 6], np.float32),
                 np.array([1, 1], np.float32), np.array([0, 0], np.float32), 3, 2)
    ok &= _check("example 2", np.array([0, 1, 2, 3], np.float32),
                 np.array([2.0, 0.5], np.float32),
                 np.array([1.0, -1.0], np.float32), 2, 2)
    for N, C in ((1, 1), (1, 7), (5, 3), (17, 33), (1000, 64), (10000, 5),
                 (3, 1024)):
        x = rng.uniform(-100, 100, N * C).astype(np.float32)
        g = rng.uniform(0.1, 10, C).astype(np.float32)
        b = rng.uniform(-10, 10, C).astype(np.float32)
        ok &= _check(f"{N}x{C}", x, g, b, N, C)

    # Benchmark: N=5000, C in {128, 512, 1024}.
    import statistics
    import time

    N = 5000
    for C in (128, 512, 1024):
        x = rng.uniform(-100, 100, N * C).astype(np.float32)
        g = rng.uniform(0.1, 10, C).astype(np.float32)
        b = rng.uniform(-10, 10, C).astype(np.float32)
        ok &= _check("benchmark", x, g, b, N, C)
        dx = cuda.to_device(x)
        dg = cuda.to_device(g)
        db = cuda.to_device(b)
        dy = cuda.device_array(N * C, dtype=np.float32)
        solve(dx, dg, db, dy, N, C, 1e-5)  # warm-up / JIT
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            solve(dx, dg, db, dy, N, C, 1e-5)
            times.append((time.perf_counter() - t0) * 1000.0)
        ms = statistics.median(times)
        print(f"Perf N={N} C={C}: {ms:.4f} ms, "
              f"{3 * N * C * 4 / (ms * 1e6):.1f} GB/s")
    print("ALL OK" if ok else "FAILURES")
