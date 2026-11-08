"""Numba port of ../cuda-basics/max-of-array/max-of-array.cu

Numba supports cuda.atomic.max on float32 directly, so the atomicCAS
emulation from the CUDA C version is not needed.
"""

import math

import numpy as np
from numba import cuda, float32


@cuda.jit(device=True)
def _warp_max(val):
    for off in (16, 8, 4, 2, 1):
        other = cuda.shfl_down_sync(0xFFFFFFFF, val, off)
        if other > val:
            val = other
    return val


@cuda.jit
def max_kernel(input, result, N):
    val = -math.inf
    i = cuda.grid(1)
    stride = cuda.gridDim.x * cuda.blockDim.x
    for idx in range(i, N, stride):
        if input[idx] > val:
            val = input[idx]

    val = _warp_max(val)
    warp_max = cuda.shared.array(32, float32)
    lane = cuda.threadIdx.x & 31
    warp = cuda.threadIdx.x >> 5
    if lane == 0:
        warp_max[warp] = val
    cuda.syncthreads()

    if warp == 0:
        num_warps = (cuda.blockDim.x + 31) >> 5
        val = warp_max[lane] if lane < num_warps else -math.inf
        val = _warp_max(val)
        if lane == 0:
            cuda.atomic.max(result, 0, val)


def solve(input, result, N):
    threads = 256
    blocks = (N + threads - 1) // threads
    result[0] = -math.inf
    max_kernel[blocks, threads](input, result, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N = 100_003
    x = rng.standard_normal(N, dtype=np.float32)
    x[rng.integers(N)] = 9.5
    result = cuda.device_array(1, dtype=np.float32)
    solve(cuda.to_device(x), result, N)
    np.testing.assert_allclose(result.copy_to_host()[0], x.max(), rtol=1e-6)
    print("PASS max-of-array")


if __name__ == "__main__":
    test()
