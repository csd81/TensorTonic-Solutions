"""Numba port of ../cuda-basics/matrix-transpose/matrix-transpose.cu"""

import numpy as np
from numba import cuda, float32

TILE = 16
TILE_PAD = TILE + 1


@cuda.jit
def matrix_transpose_kernel(A, B, M, N):
    tile = cuda.shared.array((TILE, TILE_PAD), float32)  # +1 pad: no bank conflicts

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    i = cuda.blockIdx.y * TILE + ty
    j = cuda.blockIdx.x * TILE + tx
    if i < M and j < N:
        tile[ty, tx] = A[i * N + j]
    else:
        tile[ty, tx] = 0.0

    cuda.syncthreads()

    bi = cuda.blockIdx.y * TILE + tx
    bj = cuda.blockIdx.x * TILE + ty
    if bi < M and bj < N:
        B[bj * M + bi] = tile[tx, ty]


def solve(A, B, M, N):
    threads = (TILE, TILE)
    blocks = ((N + TILE - 1) // TILE, (M + TILE - 1) // TILE)
    matrix_transpose_kernel[blocks, threads](A, B, M, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    M, N = 37, 53
    A = rng.standard_normal((M, N), dtype=np.float32)
    B = np.empty((N, M), dtype=np.float32)
    dA = cuda.to_device(A.reshape(-1))
    dB = cuda.to_device(B.reshape(-1))
    solve(dA, dB, M, N)
    np.testing.assert_allclose(dB.copy_to_host().reshape(N, M), A.T, rtol=1e-6, atol=1e-6)
    print("PASS matrix-transpose")


if __name__ == "__main__":
    test()
