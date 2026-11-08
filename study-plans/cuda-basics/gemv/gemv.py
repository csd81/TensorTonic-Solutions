"""Numba port of ../cuda-basics/gemv/gemv.cu"""

import numpy as np
from numba import cuda, float32


@cuda.jit
def gemv_shared_kernel(A, x, y, M, N):
    x_shared = cuda.shared.array(0, float32)  # size supplied at launch

    t = cuda.threadIdx.x
    j = t
    while j < N:
        x_shared[j] = x[j]
        j += cuda.blockDim.x
    cuda.syncthreads()

    warp_id = t // 32
    lane = t % 32
    warps_per_block = cuda.blockDim.x // 32
    row = cuda.blockIdx.x * warps_per_block + warp_id

    if row < M:
        partial = 0.0
        j = lane
        while j < N:
            partial += A[row * N + j] * x_shared[j]
            j += 32

        for off in (16, 8, 4, 2, 1):
            partial += cuda.shfl_down_sync(0xFFFFFFFF, partial, off)

        if lane == 0:
            y[row] = partial


def solve(A, x, y, M, N):
    threads = 256
    warps_per_block = threads // 32
    blocks = (M + warps_per_block - 1) // warps_per_block
    shared_bytes = N * 4
    gemv_shared_kernel[blocks, threads, 0, shared_bytes](A, x, y, M, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    M, N = 37, 53
    A = rng.standard_normal((M, N), dtype=np.float32)
    x = rng.standard_normal(N, dtype=np.float32)
    y = np.empty(M, dtype=np.float32)
    dA = cuda.to_device(A.reshape(-1))
    dx = cuda.to_device(x)
    dy = cuda.to_device(y)
    solve(dA, dx, dy, M, N)
    ref = A.astype(np.float64) @ x.astype(np.float64)
    np.testing.assert_allclose(dy.copy_to_host(), ref, rtol=1e-5, atol=1e-5)
    print("PASS gemv")


if __name__ == "__main__":
    test()
