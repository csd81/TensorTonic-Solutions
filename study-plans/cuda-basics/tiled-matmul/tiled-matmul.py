"""Numba port of ../cuda-basics/tiled-matmul/tiled-matmul.cu

Same tiled algorithm as matrix-multiplication.cu; kept as its own module
because the repository treats them as separate problems.
"""

import numpy as np
from numba import cuda, float32

TILE_DIM = 16


@cuda.jit
def tiled_matmul_kernel(A, B, C, M, N, K):
    s_A = cuda.shared.array((TILE_DIM, TILE_DIM), float32)
    s_B = cuda.shared.array((TILE_DIM, TILE_DIM), float32)

    row = cuda.blockIdx.y * TILE_DIM + cuda.threadIdx.y
    col = cuda.blockIdx.x * TILE_DIM + cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tx = cuda.threadIdx.x

    acc = 0.0
    num_tiles = (K + TILE_DIM - 1) // TILE_DIM

    for t in range(num_tiles):
        a_col = t * TILE_DIM + tx
        if row < M and a_col < K:
            s_A[ty, tx] = A[row * K + a_col]
        else:
            s_A[ty, tx] = 0.0

        b_row = t * TILE_DIM + ty
        if b_row < K and col < N:
            s_B[ty, tx] = B[b_row * N + col]
        else:
            s_B[ty, tx] = 0.0

        cuda.syncthreads()

        for k in range(TILE_DIM):
            acc += s_A[ty, k] * s_B[k, tx]

        cuda.syncthreads()

    if row < M and col < N:
        C[row * N + col] = acc


def solve(A, B, C, M, N, K):
    threads = (TILE_DIM, TILE_DIM)
    blocks = ((N + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM)
    tiled_matmul_kernel[blocks, threads](A, B, C, M, N, K)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(1)
    M, N, K = 33, 47, 21
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)
    C = np.empty((M, N), dtype=np.float32)
    dA = cuda.to_device(A.reshape(-1))
    dB = cuda.to_device(B.reshape(-1))
    dC = cuda.to_device(C.reshape(-1))
    solve(dA, dB, dC, M, N, K)
    ref = A.astype(np.float64) @ B.astype(np.float64)
    np.testing.assert_allclose(dC.copy_to_host().reshape(M, N), ref, rtol=1e-4, atol=1e-4)
    print("PASS tiled-matmul")


if __name__ == "__main__":
    test()
