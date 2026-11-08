"""Numba port of ../cuda-basics/swish/swish.cu

Scalar equivalent of the original 8-floats-per-thread kernel; __expf and
__frcp_rn become math.exp and plain division.
"""

import math

import numpy as np
from numba import cuda


@cuda.jit
def swish_kernel(input, output, N):
    i = cuda.grid(1)
    stride = cuda.gridDim.x * cuda.blockDim.x
    for idx in range(i, N, stride):
        x = input[idx]
        output[idx] = x / (1.0 + math.exp(-x))


def solve(input, output, N):
    threads = 256
    blocks = min((N + threads - 1) // threads, 160)
    swish_kernel[blocks, threads](input, output, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N = 10_003
    x = rng.standard_normal(N, dtype=np.float32) * 3
    out = np.empty_like(x)
    dx, dout = cuda.to_device(x), cuda.to_device(out)
    solve(dx, dout, N)
    xd = x.astype(np.float64)
    ref = xd / (1.0 + np.exp(-xd))
    np.testing.assert_allclose(dout.copy_to_host(), ref, rtol=1e-5, atol=1e-6)
    print("PASS swish")


if __name__ == "__main__":
    test()
