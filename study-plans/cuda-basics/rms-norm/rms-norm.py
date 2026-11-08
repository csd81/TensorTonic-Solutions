"""Numba port of ../cuda-basics/rms-norm/rms-norm.cu

The original ships float4 and scalar kernels; Numba has no float4, so this
is the scalar algorithm (one block per row, shared-memory block reduction).
"""

import math

import numpy as np
from numba import cuda, float32


@cuda.jit
def rms_norm_kernel(input, gamma, output, M, N, eps):
    tid = cuda.threadIdx.x
    row = cuda.blockIdx.x
    if row >= M:
        return

    row_offset = row * N
    local_sq = 0.0
    j = tid
    while j < N:
        val = input[row_offset + j]
        local_sq += val * val
        j += cuda.blockDim.x

    sdata = cuda.shared.array(256, float32)
    s_inv_rms = cuda.shared.array(1, float32)

    sdata[tid] = local_sq
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            sdata[tid] += sdata[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        s_inv_rms[0] = 1.0 / math.sqrt((sdata[0] / N) + eps)
    cuda.syncthreads()

    inv_rms = s_inv_rms[0]
    j = tid
    while j < N:
        output[row_offset + j] = input[row_offset + j] * inv_rms * gamma[j]
        j += cuda.blockDim.x


def solve(input, gamma, output, M, N, eps):
    threads = 256
    rms_norm_kernel[M, threads](input, gamma, output, M, N, eps)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    for N in (36, 37):  # divisible by 4 and not, to mirror the two C paths
        M, eps = 5, 1e-5
        x = rng.standard_normal((M, N), dtype=np.float32)
        gamma = rng.standard_normal(N, dtype=np.float32)
        out = np.empty_like(x)
        dx = cuda.to_device(x.reshape(-1))
        dg = cuda.to_device(gamma)
        dout = cuda.to_device(out.reshape(-1))
        solve(dx, dg, dout, M, N, eps)
        got = dout.copy_to_host().reshape(M, N)

        xd = x.astype(np.float64)
        ref = xd / np.sqrt((xd * xd).mean(axis=1, keepdims=True) + eps) * gamma
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)
    print("PASS rms-norm")


if __name__ == "__main__":
    test()
