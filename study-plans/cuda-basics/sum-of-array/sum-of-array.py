"""Numba port of ../cuda-basics/sum-of-array/sum-of-array.cu"""

import numpy as np
from numba import cuda, float32


@cuda.jit(device=True)
def _warp_sum(val):
    for off in (16, 8, 4, 2, 1):
        val += cuda.shfl_down_sync(0xFFFFFFFF, val, off)
    return val


@cuda.jit
def sum_kernel(input, result, N):
    total = 0.0
    i = cuda.grid(1)
    stride = cuda.gridDim.x * cuda.blockDim.x
    for idx in range(i, N, stride):
        total += input[idx]

    val = _warp_sum(total)
    warp_sums = cuda.shared.array(32, float32)
    lane = cuda.threadIdx.x & 31
    warp = cuda.threadIdx.x >> 5
    if lane == 0:
        warp_sums[warp] = val
    cuda.syncthreads()

    if warp == 0:
        num_warps = (cuda.blockDim.x + 31) >> 5
        val = warp_sums[lane] if lane < num_warps else 0.0
        val = _warp_sum(val)
        if lane == 0:
            cuda.atomic.add(result, 0, val)


def solve(input, result, N):
    threads = 256
    blocks = (N + threads - 1) // threads
    result[0] = 0.0
    sum_kernel[blocks, threads](input, result, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N = 100_003
    x = rng.standard_normal(N, dtype=np.float32)
    result = cuda.device_array(1, dtype=np.float32)
    solve(cuda.to_device(x), result, N)
    ref = x.astype(np.float64).sum()
    np.testing.assert_allclose(result.copy_to_host()[0], ref, rtol=1e-4, atol=1e-4)
    print("PASS sum-of-array")


if __name__ == "__main__":
    test()
