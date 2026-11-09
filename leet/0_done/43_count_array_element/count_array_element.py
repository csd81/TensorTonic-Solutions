"""Numba CUDA port of count_array_element.cu.

Count the occurrences of K in an int32 array of length N.  Pure streaming:
400 MB of reads at N = 100,000,000, so the kernel uses a grid-stride loop
over the whole array, a shuffle reduction per block, and one atomicAdd per
block into output[0].

solve(input, output, N, K) keeps the challenge signature; output is an int32
device array of at least one element.
"""

import numpy as np
from numba import cuda, int32

THREADS = 256
MAX_BLOCKS = 4096
NWARPS = THREADS // 32


@cuda.jit
def zero_kernel(output):
    output[0] = 0


@cuda.jit
def count_kernel(input, output, N, K):
    s_warp = cuda.shared.array(shape=NWARPS, dtype=int32)
    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    stride = cuda.gridDim.x * cuda.blockDim.x

    count = 0
    i = cuda.blockIdx.x * cuda.blockDim.x + tid
    while i < N:
        if input[i] == K:
            count += 1
        i += stride

    for off in (16, 8, 4, 2, 1):
        count += cuda.shfl_down_sync(0xFFFFFFFF, count, off)
    if lane == 0:
        s_warp[warp] = count
    cuda.syncthreads()

    if warp == 0:
        v = 0
        if lane < NWARPS:
            v = s_warp[lane]
        for off in (16, 8, 4, 2, 1):
            v += cuda.shfl_down_sync(0xFFFFFFFF, v, off)
        if lane == 0:
            cuda.atomic.add(output, 0, v)


def solve(input, output, N, K):
    if input is None or output is None:
        return
    if N <= 0:
        raise ValueError("N must be positive")

    blocks = (N + THREADS - 1) // THREADS
    if blocks > MAX_BLOCKS:
        blocks = MAX_BLOCKS
    if blocks < 1:
        blocks = 1

    zero_kernel[1, 1](output)
    count_kernel[blocks, THREADS](input, output, N, np.int32(K))
    cuda.synchronize()


def _run(name, data, K):
    dIn = cuda.to_device(data)
    dOut = cuda.device_array(1, dtype=np.int32)
    solve(dIn, dOut, data.size, K)
    got = int(dOut.copy_to_host()[0])
    ref = int(np.count_nonzero(data == K))
    ok = got == ref
    print(f"{name:>20} N={data.size} K={K}: got={got} expected={ref} "
          f"{'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    ok &= _run("example 1", np.array([1, 2, 3, 4, 1], dtype=np.int32), 1)
    ok &= _run("example 2", np.array([5, 10, 5, 2], dtype=np.int32), 11)
    ok &= _run("single match", np.array([7], dtype=np.int32), 7)
    ok &= _run("single miss", np.array([7], dtype=np.int32), 8)
    ok &= _run("odd count", np.array([4, 4, 4, 4, 4], dtype=np.int32), 4)

    data = rng.integers(1, 100001, 1023, dtype=np.int32)
    data[100] = 12345
    ok &= _run("below one block", data, 12345)

    data = rng.integers(1, 100001, 1024, dtype=np.int32)
    data[0] = data[1023] = 999
    ok &= _run("exact block", data, 999)

    data = rng.integers(1, 100001, 100000, dtype=np.int32)
    data[::7] = 501010
    ok &= _run("100k sparse", data, 501010)

    data = rng.integers(1, 100001, 1000003, dtype=np.int32)
    ok &= _run("1M", data, 1)

    data = np.full(5000000, 12345, dtype=np.int32)
    ok &= _run("5M all match", data, 12345)

    # Performance: N = 100,000,000, K = 501,010, median of 10 runs.
    import statistics
    import time

    N = 100000000
    K = 501010
    data = rng.integers(1, 100001, N, dtype=np.int32)
    dIn = cuda.to_device(data)
    dOut = cuda.device_array(1, dtype=np.int32)
    solve(dIn, dOut, N, K)  # warm-up
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(dIn, dOut, N, K)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    got = int(dOut.copy_to_host()[0])
    ref = int(np.count_nonzero(data == K))
    gbps = N * 4 / (ms * 1e6)
    print(f"Perf N={N} K={K}: {ms:.3f} ms, {gbps:.1f} GB/s, "
          f"got={got} expected={ref} {'OK' if got == ref else 'FAIL'}")
