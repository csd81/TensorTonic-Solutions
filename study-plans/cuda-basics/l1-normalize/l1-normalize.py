"""Numba port of ../cuda-basics/l1-normalize/l1-normalize.cu"""

import numpy as np
from numba import cuda, float32


@cuda.jit(device=True)
def _warp_sum(val):
    for off in (16, 8, 4, 2, 1):
        val += cuda.shfl_down_sync(0xFFFFFFFF, val, off)
    return val


@cuda.jit
def l1_reduce_kernel(input, sum_out, N):
    total = 0.0
    i = cuda.grid(1)
    stride = cuda.gridDim.x * cuda.blockDim.x
    for idx in range(i, N, stride):
        x = input[idx]
        total += x if x >= 0.0 else -x

    val = _warp_sum(total)
    warp_sums = cuda.shared.array(32, float32)
    lane = cuda.threadIdx.x & 31
    warp = cuda.threadIdx.x >> 5
    if lane == 0:
        warp_sums[warp] = val
    cuda.syncthreads()

    if warp == 0:
        num_warps = (cuda.blockDim.x + 31) >> 5
        val = warp_sums[lane] if lane < num_warps else 0.0
        val = _warp_sum(val)
        if lane == 0:
            cuda.atomic.add(sum_out, 0, val)


@cuda.jit
def l1_map_kernel(input, output, sum_out, N):
    inv_sum = 1.0 / sum_out[0]
    i = cuda.grid(1)
    stride = cuda.gridDim.x * cuda.blockDim.x
    for idx in range(i, N, stride):
        output[idx] = input[idx] * inv_sum


def solve(input, output, N):
    threads = 256
    num_sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = min(num_sms * 32, (N + threads - 1) // threads)
    blocks = max(blocks, 1)
    scratch = cuda.to_device(np.zeros(1, dtype=np.float32))
    l1_reduce_kernel[blocks, threads](input, scratch, N)
    l1_map_kernel[blocks, threads](input, output, scratch, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N = 50_003
    x = rng.standard_normal(N, dtype=np.float32)
    out = np.empty_like(x)
    dout = cuda.to_device(out)
    solve(cuda.to_device(x), dout, N)
    out = dout.copy_to_host()
    ref = x / np.abs(x.astype(np.float64)).sum()
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-7)
    print("PASS l1-normalize")


if __name__ == "__main__":
    test()
