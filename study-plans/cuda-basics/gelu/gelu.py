"""Numba port of ../cuda-basics/gelu/gelu.cu

Scalar equivalent of the original float4 kernel. erff comes from CUDA's
libdevice (numba.cuda.libdevice), Numba's replacement for the device intrinsic.
"""

import math

import numpy as np
from numba import cuda, float32
from numba.cuda import libdevice


@cuda.jit
def gelu_kernel(input, output, N):
    i = cuda.grid(1)
    stride = cuda.gridDim.x * cuda.blockDim.x
    for idx in range(i, N, stride):
        x = input[idx]
        output[idx] = 0.5 * x * (1.0 + libdevice.erff(x * float32(0.7071067811865476)))


def solve(input, output, N):
    threads = 256
    blocks = min((N + threads - 1) // threads, 2560)
    gelu_kernel[blocks, threads](input, output, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N = 10_003
    x = rng.standard_normal(N, dtype=np.float32) * 2
    out = np.empty_like(x)
    dx, dout = cuda.to_device(x), cuda.to_device(out)
    solve(dx, dout, N)
    xd = x.astype(np.float64)
    ref = 0.5 * xd * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in xd]))
    np.testing.assert_allclose(dout.copy_to_host(), ref, rtol=1e-5, atol=1e-6)
    print("PASS gelu")


if __name__ == "__main__":
    test()
