"""Numba port of ../cuda-basics/relu/relu.cu

The original vectorizes with float4; Numba has no float4, so this is the
scalar equivalent (same math, one element per grid-stride iteration).
"""

import numpy as np
from numba import cuda


@cuda.jit
def relu_kernel(input, output, N):
    i = cuda.grid(1)
    stride = cuda.gridDim.x * cuda.blockDim.x
    for idx in range(i, N, stride):
        x = input[idx]
        output[idx] = x if x > 0.0 else 0.0


def solve(input, output, N):
    threads = 256
    blocks = (N + threads - 1) // threads
    relu_kernel[blocks, threads](input, output, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N = 10_003
    x = rng.standard_normal(N, dtype=np.float32)
    out = np.empty_like(x)
    dx, dout = cuda.to_device(x), cuda.to_device(out)
    solve(dx, dout, N)
    np.testing.assert_allclose(dout.copy_to_host(), np.maximum(x, 0), rtol=1e-6, atol=1e-6)
    print("PASS relu")


if __name__ == "__main__":
    test()
