"""Numba port of ../cuda-basics/softmax/softmax.cu

Single-block, three-pass numerically stable softmax with warp shuffles,
mirroring the original 1024-thread launch.
"""

import math

import numpy as np
from numba import cuda, float32


@cuda.jit
def softmax_kernel(input, output, N):
    s_max = cuda.shared.array(32, float32)
    s_sum = cuda.shared.array(32, float32)
    global_max = cuda.shared.array(1, float32)
    global_sum = cuda.shared.array(1, float32)

    tid = cuda.threadIdx.x
    lane = tid % 32
    wid = tid // 32

    # Pass 1: max
    local_max = input[0]
    i = tid
    while i < N:
        val = input[i]
        if val > local_max:
            local_max = val
        i += cuda.blockDim.x

    for off in (16, 8, 4, 2, 1):
        other = cuda.shfl_down_sync(0xFFFFFFFF, local_max, off)
        if other > local_max:
            local_max = other

    if lane == 0:
        s_max[wid] = local_max
    cuda.syncthreads()

    if wid == 0:
        num_warps = cuda.blockDim.x // 32
        val = s_max[lane] if lane < num_warps else s_max[0]
        for off in (16, 8, 4, 2, 1):
            other = cuda.shfl_down_sync(0xFFFFFFFF, val, off)
            if other > val:
                val = other
        if lane == 0:
            global_max[0] = val
    cuda.syncthreads()

    m = global_max[0]

    # Pass 2: sum of exp(x - max)
    local_sum = 0.0
    i = tid
    while i < N:
        local_sum += math.exp(input[i] - m)
        i += cuda.blockDim.x

    for off in (16, 8, 4, 2, 1):
        local_sum += cuda.shfl_down_sync(0xFFFFFFFF, local_sum, off)

    if lane == 0:
        s_sum[wid] = local_sum
    cuda.syncthreads()

    if wid == 0:
        num_warps = cuda.blockDim.x // 32
        val = s_sum[lane] if lane < num_warps else 0.0
        for off in (16, 8, 4, 2, 1):
            val += cuda.shfl_down_sync(0xFFFFFFFF, val, off)
        if lane == 0:
            global_sum[0] = val
    cuda.syncthreads()

    s = global_sum[0]

    # Pass 3: normalize
    i = tid
    while i < N:
        output[i] = math.exp(input[i] - m) / s
        i += cuda.blockDim.x


def solve(input, output, N):
    softmax_kernel[1, 1024](input, output, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N = 5009
    x = rng.standard_normal(N, dtype=np.float32) * 10
    out = np.empty_like(x)
    dx, dout = cuda.to_device(x), cuda.to_device(out)
    solve(dx, dout, N)

    xd = x.astype(np.float64)
    e = np.exp(xd - xd.max())
    ref = e / e.sum()
    np.testing.assert_allclose(dout.copy_to_host(), ref, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(dout.copy_to_host().sum(), 1.0, rtol=1e-6)
    print("PASS softmax")


if __name__ == "__main__":
    test()
