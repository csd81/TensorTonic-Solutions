"""Numba port of ../cuda-basics/layer-norm/layer-norm.cu"""

import math

import numpy as np
from numba import cuda, float32


@cuda.jit
def layer_norm_kernel(input, gamma, beta, output, M, N, eps):
    tid = cuda.threadIdx.x
    row = cuda.blockIdx.x
    if row >= M:
        return

    row_offset = row * N
    local_sum = 0.0
    local_sq = 0.0
    j = tid
    while j < N:
        val = input[row_offset + j]
        local_sum += val
        local_sq += val * val
        j += cuda.blockDim.x

    s_sum = cuda.shared.array(256, float32)
    s_sq = cuda.shared.array(256, float32)
    s_mean = cuda.shared.array(1, float32)
    s_inv_std = cuda.shared.array(1, float32)

    s_sum[tid] = local_sum
    s_sq[tid] = local_sq
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            s_sum[tid] += s_sum[tid + stride]
            s_sq[tid] += s_sq[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        mean = s_sum[0] / N
        var = (s_sq[0] / N) - (mean * mean)
        if var < 0.0:
            var = 0.0
        s_mean[0] = mean
        s_inv_std[0] = 1.0 / math.sqrt(var + eps)
    cuda.syncthreads()

    mean = s_mean[0]
    inv_std = s_inv_std[0]
    j = tid
    while j < N:
        val = input[row_offset + j]
        output[row_offset + j] = (val - mean) * inv_std * gamma[j] + beta[j]
        j += cuda.blockDim.x


def solve(input, gamma, beta, output, M, N, eps):
    threads = 256
    layer_norm_kernel[M, threads](input, gamma, beta, output, M, N, eps)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    M, N, eps = 7, 300, 1e-5
    x = rng.standard_normal((M, N), dtype=np.float32)
    gamma = rng.standard_normal(N, dtype=np.float32)
    beta = rng.standard_normal(N, dtype=np.float32)
    out = np.empty_like(x)
    dx = cuda.to_device(x.reshape(-1))
    dg, db = cuda.to_device(gamma), cuda.to_device(beta)
    dout = cuda.to_device(out.reshape(-1))
    solve(dx, dg, db, dout, M, N, eps)
    got = dout.copy_to_host().reshape(M, N)

    xd = x.astype(np.float64)
    mu = xd.mean(axis=1, keepdims=True)
    var = xd.var(axis=1, keepdims=True)
    ref = (xd - mu) / np.sqrt(var + eps) * gamma + beta
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)
    print("PASS layer-norm")


if __name__ == "__main__":
    test()
