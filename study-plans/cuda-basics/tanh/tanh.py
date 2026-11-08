"""Numba port of ../cuda-basics/tanh/tanh.cu

Scalar equivalent of the original float4 grid-stride kernel.
"""

import math

import numpy as np
from numba import cuda


@cuda.jit
def tanh_kernel(input, output, N):
    i = cuda.grid(1)
    stride = cuda.gridDim.x * cuda.blockDim.x
    for idx in range(i, N, stride):
        output[idx] = math.tanh(input[idx])


def solve(input, output, N):
    threads = 256
    blocks = min((N + threads - 1) // threads, 512)
    tanh_kernel[blocks, threads](input, output, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N = 10_003
    x = rng.standard_normal(N, dtype=np.float32) * 2
    out = np.empty_like(x)
    dx, dout = cuda.to_device(x), cuda.to_device(out)
    solve(dx, dout, N)
    np.testing.assert_allclose(dout.copy_to_host(), np.tanh(x.astype(np.float64)), rtol=1e-6, atol=1e-6)
    print("PASS tanh")


if __name__ == "__main__":
    test()
