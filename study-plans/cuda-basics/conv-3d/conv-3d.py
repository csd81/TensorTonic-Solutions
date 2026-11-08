"""Numba port of ../cuda-basics/conv-3d/conv-3d.cu"""

import numpy as np
from numba import cuda


@cuda.jit
def conv3d_kernel(input, kernel, output, D, H, W, kD, kH, kW):
    out_d = D - kD + 1
    out_h = H - kH + 1
    out_w = W - kW + 1

    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # width
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y  # height
    d = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z  # depth

    if d < out_d and i < out_h and j < out_w:
        total = 0.0
        for a in range(kD):
            for b in range(kH):
                for c in range(kW):
                    in_idx = (d + a) * H * W + (i + b) * W + (j + c)
                    k_idx = a * kH * kW + b * kW + c
                    total += input[in_idx] * kernel[k_idx]
        out_idx = d * out_h * out_w + i * out_w + j
        output[out_idx] = total


def solve(input, kernel, output, D, H, W, kD, kH, kW):
    out_d = D - kD + 1
    out_h = H - kH + 1
    out_w = W - kW + 1
    threads = (8, 8, 8)
    blocks = ((out_w + 7) // 8, (out_h + 7) // 8, (out_d + 7) // 8)
    conv3d_kernel[blocks, threads](input, kernel, output, D, H, W, kD, kH, kW)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    D, H, W, kD, kH, kW = 7, 8, 9, 2, 3, 2
    x = rng.standard_normal((D, H, W), dtype=np.float32)
    k = rng.standard_normal((kD, kH, kW), dtype=np.float32)
    out_d, out_h, out_w = D - kD + 1, H - kH + 1, W - kW + 1
    out = np.empty((out_d, out_h, out_w), dtype=np.float32)

    dx = cuda.to_device(x.reshape(-1))
    dk = cuda.to_device(k.reshape(-1))
    dout = cuda.to_device(out.reshape(-1))
    solve(dx, dk, dout, D, H, W, kD, kH, kW)

    windows = np.lib.stride_tricks.sliding_window_view(x.astype(np.float64), (kD, kH, kW))
    ref = np.einsum("abcdef,def->abc", windows, k.astype(np.float64))
    np.testing.assert_allclose(dout.copy_to_host().reshape(out_d, out_h, out_w), ref, rtol=1e-5, atol=1e-5)
    print("PASS conv-3d")


if __name__ == "__main__":
    test()
