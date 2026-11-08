"""Numba port of ../cuda-basics/argmax/argmax.cu"""

import math

import numpy as np
from numba import cuda, float32, int32

INT_MAX = 2147483647


@cuda.jit
def argmax_kernel(input, block_vals, block_idxs, N):
    s_vals = cuda.shared.array(32, float32)
    s_idxs = cuda.shared.array(32, int32)

    tid = cuda.threadIdx.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + tid

    if idx < N:
        val = input[idx]
        cur_idx = idx
    else:
        val = -math.inf
        cur_idx = INT_MAX

    for off in (16, 8, 4, 2, 1):
        val2 = cuda.shfl_down_sync(0xFFFFFFFF, val, off)
        idx2 = cuda.shfl_down_sync(0xFFFFFFFF, cur_idx, off)
        if val2 > val or (val2 == val and idx2 < cur_idx):
            val = val2
            cur_idx = idx2

    lane = tid % 32
    warp = tid // 32
    if lane == 0:
        s_vals[warp] = val
        s_idxs[warp] = cur_idx
    cuda.syncthreads()

    if warp == 0:
        num_warps = cuda.blockDim.x // 32
        if lane < num_warps:
            val = s_vals[lane]
            cur_idx = s_idxs[lane]
        else:
            val = -math.inf
            cur_idx = INT_MAX

        for off in (16, 8, 4, 2, 1):
            val2 = cuda.shfl_down_sync(0xFFFFFFFF, val, off)
            idx2 = cuda.shfl_down_sync(0xFFFFFFFF, cur_idx, off)
            if val2 > val or (val2 == val and idx2 < cur_idx):
                val = val2
                cur_idx = idx2

        if lane == 0:
            block_vals[cuda.blockIdx.x] = val
            block_idxs[cuda.blockIdx.x] = cur_idx


@cuda.jit
def argmax_finalize_kernel(block_vals, block_idxs, result, num_blocks):
    s_vals = cuda.shared.array(32, float32)
    s_idxs = cuda.shared.array(32, int32)

    tid = cuda.threadIdx.x
    my_val = -math.inf
    my_idx = INT_MAX

    i = tid
    while i < num_blocks:
        val = block_vals[i]
        idx = block_idxs[i]
        if val > my_val or (val == my_val and idx < my_idx):
            my_val = val
            my_idx = idx
        i += cuda.blockDim.x

    for off in (16, 8, 4, 2, 1):
        val2 = cuda.shfl_down_sync(0xFFFFFFFF, my_val, off)
        idx2 = cuda.shfl_down_sync(0xFFFFFFFF, my_idx, off)
        if val2 > my_val or (val2 == my_val and idx2 < my_idx):
            my_val = val2
            my_idx = idx2

    lane = tid % 32
    warp = tid // 32
    if lane == 0:
        s_vals[warp] = my_val
        s_idxs[warp] = my_idx
    cuda.syncthreads()

    if warp == 0:
        num_warps = cuda.blockDim.x // 32
        if lane < num_warps:
            my_val = s_vals[lane]
            my_idx = s_idxs[lane]
        else:
            my_val = -math.inf
            my_idx = INT_MAX

        for off in (16, 8, 4, 2, 1):
            val2 = cuda.shfl_down_sync(0xFFFFFFFF, my_val, off)
            idx2 = cuda.shfl_down_sync(0xFFFFFFFF, my_idx, off)
            if val2 > my_val or (val2 == my_val and idx2 < my_idx):
                my_val = val2
                my_idx = idx2

        if lane == 0:
            result[0] = my_idx


def solve(input, result, N):
    threads = 256
    blocks = (N + threads - 1) // threads
    block_vals = cuda.device_array(blocks, dtype=np.float32)
    block_idxs = cuda.device_array(blocks, dtype=np.int32)
    argmax_kernel[blocks, threads](input, block_vals, block_idxs, N)
    argmax_finalize_kernel[1, threads](block_vals, block_idxs, result, blocks)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N = 100_003
    x = rng.integers(0, 5, N).astype(np.float32)  # lots of ties
    x[777] = 9.0
    x[55555] = 9.0  # tie; argmax must return the lower index
    result = cuda.device_array(1, dtype=np.int32)
    solve(cuda.to_device(x), result, N)
    assert result.copy_to_host()[0] == int(np.argmax(x)), result.copy_to_host()
    print("PASS argmax")


if __name__ == "__main__":
    test()
