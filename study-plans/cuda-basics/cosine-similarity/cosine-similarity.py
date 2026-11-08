"""Numba port of ../cuda-basics/cosine-similarity/cosine-similarity.cu"""

import math

import numpy as np
from numba import cuda, float32


@cuda.jit(device=True)
def _warp_sum(val):
    for off in (16, 8, 4, 2, 1):
        val += cuda.shfl_down_sync(0xFFFFFFFF, val, off)
    return val


@cuda.jit
def cosine_partials_kernel(A, B, scratch, N):
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    i = cuda.grid(1)
    stride = cuda.gridDim.x * cuda.blockDim.x
    for idx in range(i, N, stride):
        a = A[idx]
        b = B[idx]
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    d = _warp_sum(dot)
    na = _warp_sum(norm_a)
    nb = _warp_sum(norm_b)

    warp_dot = cuda.shared.array(32, float32)
    warp_na = cuda.shared.array(32, float32)
    warp_nb = cuda.shared.array(32, float32)
    lane = cuda.threadIdx.x & 31
    warp = cuda.threadIdx.x >> 5
    if lane == 0:
        warp_dot[warp] = d
        warp_na[warp] = na
        warp_nb[warp] = nb
    cuda.syncthreads()

    if warp == 0:
        num_warps = (cuda.blockDim.x + 31) >> 5
        d = warp_dot[lane] if lane < num_warps else 0.0
        na = warp_na[lane] if lane < num_warps else 0.0
        nb = warp_nb[lane] if lane < num_warps else 0.0
        d = _warp_sum(d)
        na = _warp_sum(na)
        nb = _warp_sum(nb)
        if lane == 0:
            cuda.atomic.add(scratch, 0, d)
            cuda.atomic.add(scratch, 1, na)
            cuda.atomic.add(scratch, 2, nb)


@cuda.jit
def cosine_finalize_kernel(scratch, result):
    if cuda.threadIdx.x == 0:
        result[0] = scratch[0] / math.sqrt(scratch[1]) / math.sqrt(scratch[2])


def solve(A, B, result, N):
    threads = 256
    blocks = min((N + threads - 1) // threads, 1024)
    scratch = cuda.to_device(np.zeros(3, dtype=np.float32))
    cosine_partials_kernel[blocks, threads](A, B, scratch, N)
    cosine_finalize_kernel[1, 1](scratch, result)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N = 100_003
    A = rng.standard_normal(N, dtype=np.float32)
    B = rng.standard_normal(N, dtype=np.float32)
    result = cuda.device_array(1, dtype=np.float32)
    solve(cuda.to_device(A), cuda.to_device(B), result, N)
    Ad, Bd = A.astype(np.float64), B.astype(np.float64)
    ref = np.dot(Ad, Bd) / np.linalg.norm(Ad) / np.linalg.norm(Bd)
    np.testing.assert_allclose(result.copy_to_host()[0], ref, rtol=1e-4, atol=1e-5)
    print("PASS cosine-similarity")


if __name__ == "__main__":
    test()
