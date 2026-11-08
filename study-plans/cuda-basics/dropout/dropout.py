"""Numba port of ../cuda-basics/dropout/dropout.cu

Inverted dropout with a supplied binary mask; scalar grid-stride equivalent
of the original float4 kernel.
"""

import numpy as np
from numba import cuda


@cuda.jit
def dropout_kernel(input, mask, output, p, N):
    scale = 1.0 / (1.0 - p)
    i = cuda.grid(1)
    stride = cuda.gridDim.x * cuda.blockDim.x
    for idx in range(i, N, stride):
        output[idx] = input[idx] * mask[idx] * scale


def solve(input, mask, output, p, N):
    threads = 256
    blocks = max((N + threads - 1) // threads, 1)
    dropout_kernel[blocks, threads](input, mask, output, p, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N = 10_003
    p = 0.25
    x = rng.standard_normal(N, dtype=np.float32)
    mask = (rng.random(N) >= p).astype(np.float32)
    out = np.empty_like(x)
    dx, dm, dout = cuda.to_device(x), cuda.to_device(mask), cuda.to_device(out)
    solve(dx, dm, dout, p, N)
    ref = x * mask / (1.0 - p)
    np.testing.assert_allclose(dout.copy_to_host(), ref, rtol=1e-6, atol=1e-6)
    print("PASS dropout")


if __name__ == "__main__":
    test()
