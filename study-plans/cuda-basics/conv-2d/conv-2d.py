"""Numba port of ../cuda-basics/conv-2d/conv-2d.cu"""

import numpy as np
from numba import cuda


@cuda.jit
def conv2d_kernel(input, kernel, output, H, W, kH, kW):
    out_h = H - kH + 1
    out_w = W - kW + 1

    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if row < out_h and col < out_w:
        total = 0.0
        for i in range(kH):
            in_row = row + i
            for j in range(kW):
                in_col = col + j
                total += input[in_row * W + in_col] * kernel[i * kW + j]
        output[row * out_w + col] = total


def solve(input, kernel, output, H, W, kH, kW):
    out_h = H - kH + 1
    out_w = W - kW + 1
    threads = (16, 16)
    blocks = ((out_w + 15) // 16, (out_h + 15) // 16)
    conv2d_kernel[blocks, threads](input, kernel, output, H, W, kH, kW)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    H, W, kH, kW = 20, 25, 3, 4
    x = rng.standard_normal((H, W), dtype=np.float32)
    k = rng.standard_normal((kH, kW), dtype=np.float32)
    out_h, out_w = H - kH + 1, W - kW + 1
    out = np.empty((out_h, out_w), dtype=np.float32)

    dx = cuda.to_device(x.reshape(-1))
    dk = cuda.to_device(k.reshape(-1))
    dout = cuda.to_device(out.reshape(-1))
    solve(dx, dk, dout, H, W, kH, kW)

    windows = np.lib.stride_tricks.sliding_window_view(x.astype(np.float64), (kH, kW))
    ref = np.einsum("ijkl,kl->ij", windows, k.astype(np.float64))
    np.testing.assert_allclose(dout.copy_to_host().reshape(out_h, out_w), ref, rtol=1e-5, atol=1e-5)
    print("PASS conv-2d")


if __name__ == "__main__":
    test()
