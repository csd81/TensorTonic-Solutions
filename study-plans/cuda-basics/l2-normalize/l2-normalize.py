"""Numba port of ../cuda-basics/l2-normalize/l2-normalize.cu"""

import math

import numpy as np
from numba import cuda, float32


@cuda.jit
def reduce_sq_sum(input, sum_out, N):
    partial = 0.0
    i = cuda.grid(1)
    stride = cuda.gridDim.x * cuda.blockDim.x
    for idx in range(i, N, stride):
        x = input[idx]
        partial += x * x

    sdata = cuda.shared.array(256, float32)
    tid = cuda.threadIdx.x
    sdata[tid] = partial
    cuda.syncthreads()

    stride_ = cuda.blockDim.x // 2
    while stride_ > 0:
        if tid < stride_:
            sdata[tid] += sdata[tid + stride_]
        cuda.syncthreads()
        stride_ //= 2

    if tid == 0:
        cuda.atomic.add(sum_out, 0, sdata[0])


@cuda.jit
def divide_by_sqrt(input, output, sum_out, N):
    factor = 1.0 / math.sqrt(sum_out[0])
    i = cuda.grid(1)
    stride = cuda.gridDim.x * cuda.blockDim.x
    for idx in range(i, N, stride):
        output[idx] = input[idx] * factor


def solve(input, output, N):
    threads = 256
    blocks = (N + threads - 1) // threads
    scratch = cuda.to_device(np.zeros(1, dtype=np.float32))
    reduce_sq_sum[blocks, threads](input, scratch, N)
    divide_by_sqrt[blocks, threads](input, output, scratch, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N = 50_003
    x = rng.standard_normal(N, dtype=np.float32)
    out = np.empty_like(x)
    dout = cuda.to_device(out)
    solve(cuda.to_device(x), dout, N)
    out = dout.copy_to_host()
    ref = x / np.linalg.norm(x.astype(np.float64))
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-7)
    print("PASS l2-normalize")


if __name__ == "__main__":
    test()
