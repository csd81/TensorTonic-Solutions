"""Numba CUDA port of mean_squared_error.cu.

mse = (1/N) * sum_i (predictions[i] - targets[i])^2.  N = 50,000,000 at the
benchmark shape (400 MB of reads, read-bound; the C++ kernel measures 4.63 ms
= 86 GB/s against the starter's 5.64 ms, which pays for FP64 arithmetic in
the inner loop -- 1/32 rate on sm_61).

The port keeps the same two-stage structure: FP32 accumulation with warp
shuffles into one partial per block (Numba has no float4 loads, so the loads
are scalar at a block-strided offset, still coalesced), then a single-block
final kernel that sums the <= 2048 block partials in float64 and divides by
N.  FP64 is used only there, where its rate is irrelevant.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, float64, int32

THREADS = 256
BLOCKS_CAP = 2048

_block_sums = None


@cuda.jit(cache=True)
def mse_kernel(p, t, block_sums, N):
    ws = cuda.shared.array(8, dtype=float32)  # one slot per warp (256 thr)
    s0 = float32(0.0)
    s1 = float32(0.0)

    tid = int32(cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x)
    stride = int32(cuda.blockDim.x * cuda.gridDim.x)
    i = tid
    while i < N:
        d0 = p[i] - t[i]
        s0 += d0 * d0
        i += stride

    s = s0 + s1
    off = int32(16)
    while off > 0:
        s += cuda.shfl_down_sync(0xFFFFFFFF, s, off)
        off >>= 1
    lane = int32(cuda.threadIdx.x) & 31
    warp = int32(cuda.threadIdx.x) >> 5
    if lane == 0:
        ws[warp] = s
    cuda.syncthreads()
    if cuda.threadIdx.x == 0:
        total = float32(0.0)
        w = int32(0)
        while w < (int32(cuda.blockDim.x) + int32(31)) >> 5:
            total += ws[w]
            w += int32(1)
        block_sums[cuda.blockIdx.x] = total


@cuda.jit(cache=True)
def final_kernel(block_sums, mse, nb, N):
    shared = cuda.shared.array(256, dtype=float64)
    s = float64(0.0)
    i = int32(cuda.threadIdx.x)
    while i < nb:
        s += float64(block_sums[i])
        i += int32(256)
    shared[cuda.threadIdx.x] = s
    cuda.syncthreads()
    off = int32(128)
    while off > 0:
        if cuda.threadIdx.x < off:
            shared[cuda.threadIdx.x] += shared[cuda.threadIdx.x + off]
        cuda.syncthreads()
        off >>= 1
    if cuda.threadIdx.x == 0:
        mse[0] = float32(shared[0] / float64(N))


def solve(p, t, mse, N):
    """p, t: float32 device arrays of length N; mse: float32 device array [1]."""
    global _block_sums
    if N <= 0:
        return
    if _block_sums is None:
        _block_sums = cuda.device_array(BLOCKS_CAP, dtype=np.float32)
    blocks = min(BLOCKS_CAP, max(1, (N + THREADS - 1) // THREADS))
    mse_kernel[blocks, THREADS](p, t, _block_sums, int32(N))
    final_kernel[1, 256](_block_sums, mse, int32(blocks), int32(N))
    cuda.synchronize()


def _ref(p, t, N):
    total = 0.0
    for s in range(0, N, 1 << 22):
        e = min(N, s + (1 << 22))
        d = p[s:e].astype(np.float64) - t[s:e].astype(np.float64)
        total += float(np.sum(d * d))
    return total / N


def _check(name, p, t, N):
    ref = _ref(p, t, N)
    dp = cuda.to_device(np.ascontiguousarray(p, dtype=np.float32))
    dt = cuda.to_device(np.ascontiguousarray(t, dtype=np.float32))
    dm = cuda.device_array(1, dtype=np.float32)
    solve(dp, dt, dm, N)
    got = float(dm.copy_to_host()[0])
    ok = abs(got - ref) <= 1e-4 * abs(ref) + 1e-3
    print(f"{name:>16} N={N}: got={got:.6g} ref={ref:.6g} "
          f"err={abs(got - ref):.2e} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    ok &= _check("example 1", np.array([1, 2, 3, 4], np.float32),
                 np.array([1.5, 2.5, 3.5, 4.5], np.float32), 4)
    ok &= _check("example 2", np.array([10, 20, 30], np.float32),
                 np.array([12, 18, 33], np.float32), 3)
    for N in (1, 7, 1000, 100000, 1 << 20):
        p = rng.uniform(-1000, 1000, N).astype(np.float32)
        t = rng.uniform(-1000, 1000, N).astype(np.float32)
        ok &= _check(f"random N={N}", p, t, N)

    # Benchmark: N = 50,000,000.
    import statistics
    import time

    N = 50000000
    p = rng.uniform(-1000, 1000, N).astype(np.float32)
    t = rng.uniform(-1000, 1000, N).astype(np.float32)
    ok &= _check("benchmark shape", p, t, N)
    dp = cuda.to_device(p)
    dt = cuda.to_device(t)
    dm = cuda.device_array(1, dtype=np.float32)
    solve(dp, dt, dm, N)  # warm-up / JIT
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(dp, dt, dm, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    print(f"Perf N={N}: {ms:.4f} ms, {2 * N * 4 / (ms * 1e6):.1f} GB/s")
    print("ALL OK" if ok else "FAILURES")
