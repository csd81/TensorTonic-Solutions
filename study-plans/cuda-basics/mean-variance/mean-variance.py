"""Numba port of ../cuda-basics/mean-variance/mean-variance.cu"""

import numpy as np
from numba import cuda, float32


@cuda.jit(device=True)
def _warp_sum(val):
    for off in (16, 8, 4, 2, 1):
        val += cuda.shfl_down_sync(0xFFFFFFFF, val, off)
    return val


@cuda.jit
def mean_variance_kernel(input, sum_out, sumsq_out, N):
    local_sum = 0.0
    local_sq = 0.0
    i = cuda.grid(1)
    stride = cuda.gridDim.x * cuda.blockDim.x
    for idx in range(i, N, stride):
        x = input[idx]
        local_sum += x
        local_sq += x * x

    s = _warp_sum(local_sum)
    q = _warp_sum(local_sq)

    warp_sum = cuda.shared.array(32, float32)
    warp_sq = cuda.shared.array(32, float32)
    lane = cuda.threadIdx.x & 31
    warp = cuda.threadIdx.x >> 5
    if lane == 0:
        warp_sum[warp] = s
        warp_sq[warp] = q
    cuda.syncthreads()

    if warp == 0:
        num_warps = (cuda.blockDim.x + 31) >> 5
        s = warp_sum[lane] if lane < num_warps else 0.0
        q = warp_sq[lane] if lane < num_warps else 0.0
        s = _warp_sum(s)
        q = _warp_sum(q)
        if lane == 0:
            cuda.atomic.add(sum_out, 0, s)
            cuda.atomic.add(sumsq_out, 0, q)


@cuda.jit
def finalize_kernel(mean_out, var_out, N):
    if cuda.blockIdx.x == 0 and cuda.threadIdx.x == 0:
        total = mean_out[0]
        total_sq = var_out[0]
        mean = total / N
        var = (total_sq / N) - (mean * mean)
        if var < 0.0:
            var = 0.0
        mean_out[0] = mean
        var_out[0] = var


def solve(input, mean_out, var_out, N):
    threads = 256
    blocks = min((N + threads - 1) // threads, 1024)
    mean_out[0] = 0.0
    var_out[0] = 0.0
    mean_variance_kernel[blocks, threads](input, mean_out, var_out, N)
    finalize_kernel[1, 1](mean_out, var_out, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N = 100_003
    x = rng.standard_normal(N, dtype=np.float32) * 4 + 2
    mean = cuda.device_array(1, dtype=np.float32)
    var = cuda.device_array(1, dtype=np.float32)
    solve(cuda.to_device(x), mean, var, N)
    np.testing.assert_allclose(mean.copy_to_host()[0], x.mean(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(var.copy_to_host()[0], x.var(), rtol=1e-4, atol=1e-4)
    print("PASS mean-variance")


if __name__ == "__main__":
    test()
