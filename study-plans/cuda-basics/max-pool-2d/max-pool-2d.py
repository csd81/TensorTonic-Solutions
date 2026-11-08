"""Numba port of ../cuda-basics/max-pool-2d/max-pool-2d.cu"""

import numpy as np
from numba import cuda


@cuda.jit
def max_pool_2d_kernel(input, output, H, W, kH, kW, sH, sW):
    out_h = (H - kH) // sH + 1
    out_w = (W - kW) // sW + 1

    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if row < out_h and col < out_w:
        max_val = -np.inf
        in_row_start = row * sH
        in_col_start = col * sW
        for i in range(kH):
            for j in range(kW):
                val = input[(in_row_start + i) * W + (in_col_start + j)]
                if val > max_val:
                    max_val = val
        output[row * out_w + col] = max_val


def solve(input, output, H, W, kH, kW, sH, sW):
    out_h = (H - kH) // sH + 1
    out_w = (W - kW) // sW + 1
    threads = (16, 16)
    blocks = ((out_w + 15) // 16, (out_h + 15) // 16)
    max_pool_2d_kernel[blocks, threads](input, output, H, W, kH, kW, sH, sW)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    H, W, kH, kW, sH, sW = 20, 22, 3, 3, 2, 2
    x = rng.standard_normal((H, W), dtype=np.float32)
    out_h = (H - kH) // sH + 1
    out_w = (W - kW) // sW + 1
    out = np.empty((out_h, out_w), dtype=np.float32)

    dx = cuda.to_device(x.reshape(-1))
    dout = cuda.to_device(out.reshape(-1))
    solve(dx, dout, H, W, kH, kW, sH, sW)

    windows = np.lib.stride_tricks.sliding_window_view(x, (kH, kW))[::sH, ::sW]
    ref = windows[:out_h, :out_w].max(axis=(-1, -2))
    np.testing.assert_allclose(dout.copy_to_host().reshape(out_h, out_w), ref, rtol=1e-6)
    print("PASS max-pool-2d")


if __name__ == "__main__":
    test()
