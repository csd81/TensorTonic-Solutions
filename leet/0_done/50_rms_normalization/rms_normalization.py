"""Numba CUDA port of rms_normalization.cu.

RMS normalization: y_i = gamma * x_i / sqrt(mean(x^2) + eps) + beta.

Numba has no cooperative grid-wide barrier, so this port uses two kernels:
one writes the per-block sums of squares, and the second (each block
redundantly) adds those few dozen values, then scales and shifts.  That avoids
both the atomics and a separate zeroing launch -- kernel launches are the
dominant cost at this problem size.

solve(input, gamma, beta, output, N, eps) keeps the challenge signature.
"""

import math

import numpy as np
from numba import cuda, float32

THREADS = 256
NWARPS = THREADS // 32
MAX_BLOCKS = 128

_scratch = None


@cuda.jit
def sumsq_blocks_kernel(x, block_sums, N):
    s_warp = cuda.shared.array(shape=NWARPS, dtype=float32)
    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    stride = cuda.gridDim.x * cuda.blockDim.x

    acc = float32(0.0)
    i = cuda.blockIdx.x * cuda.blockDim.x + tid
    while i < N:
        v = x[i]
        acc += v * v
        i += stride

    for off in (16, 8, 4, 2, 1):
        acc += cuda.shfl_down_sync(0xFFFFFFFF, acc, off)
    if lane == 0:
        s_warp[warp] = acc
    cuda.syncthreads()

    if warp == 0:
        v = float32(0.0)
        if lane < NWARPS:
            v = s_warp[lane]
        for off in (16, 8, 4, 2, 1):
            v += cuda.shfl_down_sync(0xFFFFFFFF, v, off)
        if lane == 0:
            block_sums[cuda.blockIdx.x] = v


@cuda.jit
def norm_kernel(x, y, block_sums, num_blocks, gamma, beta, N, eps):
    s_scale = cuda.shared.array(shape=1, dtype=float32)
    if cuda.threadIdx.x == 0:
        total = float32(0.0)
        for b in range(num_blocks):
            total += block_sums[b]
        mean_sq = total / float32(N)
        s_scale[0] = (float32(1.0) / math.sqrt(mean_sq + eps)) * gamma
    cuda.syncthreads()
    scale = s_scale[0]

    stride = cuda.gridDim.x * cuda.blockDim.x
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    while i < N:
        y[i] = x[i] * scale + beta
        i += stride


def solve(input, gamma, beta, output, N, eps):
    global _scratch
    if input is None or output is None:
        return
    if N <= 0:
        raise ValueError("N must be positive")

    if _scratch is None:
        _scratch = cuda.device_array(MAX_BLOCKS, dtype=np.float32)

    blocks = (N + THREADS - 1) // THREADS
    if blocks > MAX_BLOCKS:
        blocks = MAX_BLOCKS
    if blocks < 1:
        blocks = 1

    sumsq_blocks_kernel[blocks, THREADS](input, _scratch, N)
    norm_kernel[blocks, THREADS](input, output, _scratch, blocks,
                                 float32(gamma), float32(beta), N,
                                 float32(eps))
    cuda.synchronize()


def _reference(x, gamma, beta, N, eps):
    xd = x.astype(np.float64)
    rms = math.sqrt(float(np.sum(xd * xd)) / N + eps)
    return xd / rms * gamma + beta


def _run(name, data, gamma, beta, eps=1e-5):
    ref = _reference(data, gamma, beta, data.size, eps)
    dIn = cuda.to_device(data)
    dOut = cuda.device_array(data.size, dtype=np.float32)
    solve(dIn, gamma, beta, dOut, data.size, eps)
    got = dOut.copy_to_host()
    max_err = float(np.max(np.abs(got - ref)))
    rel = max_err / max(1e-30, float(np.max(np.abs(ref))))
    ok = rel < 1e-5
    print(f"{name:>20} N={data.size} gamma={gamma} beta={beta}: "
          f"max_err={max_err:.3e} rel={rel:.3e} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    ok &= _run("example 1", np.array([1, 2, 3, 4], dtype=np.float32), 1.0, 0.0)
    ok &= _run("example 2", np.array([1, 2, 3], dtype=np.float32), 1.0, 0.0)
    ok &= _run("single element", np.array([5.0], dtype=np.float32), 1.0, 0.0)
    ok &= _run("single zero", np.array([0.0], dtype=np.float32), 1.0, 0.0)

    data = rng.uniform(-100.0, 100.0, 31).astype(np.float32)
    ok &= _run("odd small", data, 1.0, 0.0)
    data = rng.uniform(-100.0, 100.0, 1024).astype(np.float32)
    ok &= _run("one block", data, 2.5, -1.5)
    data = rng.uniform(-100.0, 100.0, 1025).astype(np.float32)
    ok &= _run("block + 1", data, 0.1, 10.0)
    data = rng.uniform(-100.0, 100.0, 99999).astype(np.float32)
    ok &= _run("near benchmark", data, 10.0, -10.0)
    data = rng.uniform(-100.0, 100.0, 100000).astype(np.float32)
    ok &= _run("benchmark size", data, 1.0, 0.0)
    ok &= _run("all zeros", np.zeros(4096, dtype=np.float32), 3.0, 2.0)

    # Performance: N = 100,000, median of 100 runs.
    import statistics
    import time

    N = 100000
    data = rng.uniform(-100.0, 100.0, N).astype(np.float32)
    dIn = cuda.to_device(data)
    dOut = cuda.device_array(N, dtype=np.float32)
    solve(dIn, 1.0, 0.0, dOut, N, 1e-5)  # warm-up
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        solve(dIn, 1.0, 0.0, dOut, N, 1e-5)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)

    got = dOut.copy_to_host()
    ref = _reference(data, 1.0, 0.0, N, 1e-5)
    rel = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
    bytes_moved = 3.0 * N * 4.0
    print(f"Perf N={N}: {ms * 1000:.1f} us, {bytes_moved / (ms * 1e6):.1f} GB/s, "
          f"rel={rel:.3e}, {'OK' if rel < 1e-5 else 'FAIL'}")
