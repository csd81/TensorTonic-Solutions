"""Numba port of ../cuda-basics/cross-entropy/cross-entropy.cu

One block per row; the row is staged in dynamic shared memory when
C <= 10000, matching the original's smem-size guard.
"""

import math

import numpy as np
from numba import cuda, float32


@cuda.jit
def cross_entropy_row_kernel(logits, target, partial, B, C):
    s_logits = cuda.shared.array(0, float32)
    s_warp = cuda.shared.array(32, float32)

    b = cuda.blockIdx.x
    if b >= B:
        return

    use_smem = C <= 10000
    row_offset = b * C
    tid = cuda.threadIdx.x
    lane = tid % 32
    wid = tid // 32

    # Pass 1: max
    thread_max = -math.inf
    c = tid
    while c < C:
        val = logits[row_offset + c]
        if use_smem:
            s_logits[c] = val
        if val > thread_max:
            thread_max = val
        c += cuda.blockDim.x

    val_max = thread_max
    for off in (16, 8, 4, 2, 1):
        other = cuda.shfl_down_sync(0xFFFFFFFF, val_max, off)
        if other > val_max:
            val_max = other
    if lane == 0:
        s_warp[wid] = val_max
    cuda.syncthreads()

    num_warps = (cuda.blockDim.x + 31) // 32
    row_max = s_warp[lane] if tid < num_warps else -math.inf
    if wid == 0:
        for off in (16, 8, 4, 2, 1):
            other = cuda.shfl_down_sync(0xFFFFFFFF, row_max, off)
            if other > row_max:
                row_max = other
    if tid == 0:
        s_warp[0] = row_max
    cuda.syncthreads()
    row_max = s_warp[0]

    # Pass 2: sum of exps
    thread_sum = 0.0
    c = tid
    while c < C:
        if use_smem:
            thread_sum += math.exp(s_logits[c] - row_max)
        else:
            thread_sum += math.exp(logits[row_offset + c] - row_max)
        c += cuda.blockDim.x

    val_sum = thread_sum
    for off in (16, 8, 4, 2, 1):
        val_sum += cuda.shfl_down_sync(0xFFFFFFFF, val_sum, off)
    if lane == 0:
        s_warp[wid] = val_sum
    cuda.syncthreads()

    row_sum = s_warp[lane] if tid < num_warps else 0.0
    if wid == 0:
        for off in (16, 8, 4, 2, 1):
            row_sum += cuda.shfl_down_sync(0xFFFFFFFF, row_sum, off)
    if tid == 0:
        s_warp[0] = row_sum
    cuda.syncthreads()
    row_sum = s_warp[0]

    if tid == 0:
        t = target[b]
        target_val = s_logits[t] if use_smem else logits[row_offset + t]
        row_loss = math.log(row_sum) - (target_val - row_max)
        cuda.atomic.add(partial, 0, row_loss)


@cuda.jit
def cross_entropy_finalize_kernel(loss, B):
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0 and B > 0:
        loss[0] = loss[0] / B


def solve(logits, target, loss, B, C):
    threads = 256
    loss[0] = 0.0
    shared_bytes = C * 4 if C <= 10000 else 0
    cross_entropy_row_kernel[B, threads, 0, shared_bytes](logits, target, loss, B, C)
    cross_entropy_finalize_kernel[1, 1](loss, B)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    B, C = 17, 50
    logits = rng.standard_normal((B, C), dtype=np.float32)
    target = rng.integers(0, C, B).astype(np.int32)
    loss = cuda.device_array(1, dtype=np.float32)
    solve(cuda.to_device(logits.reshape(-1)), cuda.to_device(target), loss, B, C)

    ld = logits.astype(np.float64)
    m = ld.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(ld - m).sum(axis=1))
    ref = np.mean(lse - ld[np.arange(B), target])
    np.testing.assert_allclose(loss.copy_to_host()[0], ref, rtol=1e-5, atol=1e-5)
    print("PASS cross-entropy")


if __name__ == "__main__":
    test()
