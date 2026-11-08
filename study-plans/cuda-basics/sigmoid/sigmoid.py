"""Numba port of ../cuda-basics/sigmoid/sigmoid.cu

Scalar equivalent of the original float4 grid-stride kernel.
"""

import math

import numpy as np
from numba import cuda


@cuda.jit
def sigmoid_kernel(input, output, N):
    i = cuda.grid(1)
    stride = cuda.gridDim.x * cuda.blockDim.x
    for idx in range(i, N, stride):
        output[idx] = 1.0 / (1.0 + math.exp(-input[idx]))


def solve(input, output, N):
    threads = 256
    blocks = min((N + threads - 1) // threads, 512)
    sigmoid_kernel[blocks, threads](input, output, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N = 10_003
    x = rng.standard_normal(N, dtype=np.float32) * 4
    out = np.empty_like(x)
    dx, dout = cuda.to_device(x), cuda.to_device(out)
    solve(dx, dout, N)
    ref = 1.0 / (1.0 + np.exp(-x.astype(np.float64)))
    np.testing.assert_allclose(dout.copy_to_host(), ref, rtol=1e-5, atol=1e-5)
    print("PASS sigmoid")


if __name__ == "__main__":
    test()
