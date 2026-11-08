"""Numba port of ../cuda-basics/leaky-relu/leaky-relu.cu

Scalar equivalent of the original float4 grid-stride kernel.
"""

import numpy as np
from numba import cuda


@cuda.jit
def leaky_relu_kernel(input, output, alpha, N):
    i = cuda.grid(1)
    stride = cuda.gridDim.x * cuda.blockDim.x
    for idx in range(i, N, stride):
        x = input[idx]
        output[idx] = x if x > 0.0 else alpha * x


def solve(input, output, alpha, N):
    threads = 256
    blocks = min((N + threads - 1) // threads, 1024)
    leaky_relu_kernel[blocks, threads](input, output, alpha, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N = 10_003
    alpha = 0.1
    x = rng.standard_normal(N, dtype=np.float32)
    out = np.empty_like(x)
    dx, dout = cuda.to_device(x), cuda.to_device(out)
    solve(dx, dout, alpha, N)
    ref = np.where(x > 0, x, alpha * x)
    np.testing.assert_allclose(dout.copy_to_host(), ref, rtol=1e-6, atol=1e-6)
    print("PASS leaky-relu")


if __name__ == "__main__":
    test()
