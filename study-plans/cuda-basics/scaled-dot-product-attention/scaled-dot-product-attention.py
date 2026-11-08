"""Numba port of ../cuda-basics/scaled-dot-product-attention/scaled-dot-product-attention.cu

Three stages: tiled QK^T / sqrt(D), row-wise softmax, tiled PV.
"""

import math

import numpy as np
from numba import cuda, float32

TILE = 16
TILE_PAD = 17


@cuda.jit
def scores_kernel(Q, K, scores, N, D):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    s_Q = cuda.shared.array((TILE, TILE_PAD), float32)
    s_K = cuda.shared.array((TILE, TILE_PAD), float32)

    total = 0.0
    d_start = 0
    while d_start < D:
        if row < N and d_start + tx < D:
            s_Q[ty, tx] = Q[row * D + d_start + tx]
        else:
            s_Q[ty, tx] = 0.0

        k_row = cuda.blockIdx.x * TILE + ty
        k_col = d_start + tx
        if k_row < N and k_col < D:
            s_K[ty, tx] = K[k_row * D + k_col]
        else:
            s_K[ty, tx] = 0.0

        cuda.syncthreads()

        for step in range(TILE):
            total += s_Q[ty, step] * s_K[tx, step]

        cuda.syncthreads()
        d_start += TILE

    if row < N and col < N:
        scores[row * N + col] = total / math.sqrt(D)


@cuda.jit
def softmax_rows_kernel(scores, N):
    row = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if row >= N:
        return

    base = row * N
    s_data = cuda.shared.array(256, float32)

    local_max = -math.inf
    i = tid
    while i < N:
        val = scores[base + i]
        if val > local_max:
            local_max = val
        i += cuda.blockDim.x

    s_data[tid] = local_max
    cuda.syncthreads()
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            other = s_data[tid + stride]
            if other > s_data[tid]:
                s_data[tid] = other
        cuda.syncthreads()
        stride //= 2
    row_max = s_data[0]

    local_sum = 0.0
    i = tid
    while i < N:
        val = math.exp(scores[base + i] - row_max)
        local_sum += val
        scores[base + i] = val
        i += cuda.blockDim.x

    s_data[tid] = local_sum
    cuda.syncthreads()
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            s_data[tid] += s_data[tid + stride]
        cuda.syncthreads()
        stride //= 2
    row_sum = s_data[0]

    i = tid
    while i < N:
        scores[base + i] /= row_sum
        i += cuda.blockDim.x


@cuda.jit
def av_kernel(attn, V, output, N, D):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    s_P = cuda.shared.array((TILE, TILE_PAD), float32)
    s_V = cuda.shared.array((TILE, TILE_PAD), float32)

    total = 0.0
    j_start = 0
    while j_start < N:
        if row < N and j_start + tx < N:
            s_P[ty, tx] = attn[row * N + j_start + tx]
        else:
            s_P[ty, tx] = 0.0

        v_row = j_start + ty
        v_col = cuda.blockIdx.x * TILE + tx
        if v_row < N and v_col < D:
            s_V[ty, tx] = V[v_row * D + v_col]
        else:
            s_V[ty, tx] = 0.0

        cuda.syncthreads()

        for step in range(TILE):
            total += s_P[ty, step] * s_V[step, tx]

        cuda.syncthreads()
        j_start += TILE

    if row < N and col < D:
        output[row * D + col] = total


def solve(Q, K, V, output, N, D):
    scores = cuda.device_array(N * N, dtype=np.float32)

    s_threads = (TILE, TILE)
    s_blocks = ((N + TILE - 1) // TILE, (N + TILE - 1) // TILE)
    scores_kernel[s_blocks, s_threads](Q, K, scores, N, D)

    softmax_rows_kernel[N, 256](scores, N)

    av_threads = (TILE, TILE)
    av_blocks = ((D + TILE - 1) // TILE, (N + TILE - 1) // TILE)
    av_kernel[av_blocks, av_threads](scores, V, output, N, D)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N, D = 16, 8
    Q = rng.standard_normal((N, D), dtype=np.float32)
    K = rng.standard_normal((N, D), dtype=np.float32)
    V = rng.standard_normal((N, D), dtype=np.float32)
    out = np.empty((N, D), dtype=np.float32)

    dQ, dK, dV = cuda.to_device(Q.reshape(-1)), cuda.to_device(K.reshape(-1)), cuda.to_device(V.reshape(-1))
    dout = cuda.to_device(out.reshape(-1))
    solve(dQ, dK, dV, dout, N, D)

    q, k, v = Q.astype(np.float64), K.astype(np.float64), V.astype(np.float64)
    s = q @ k.T / math.sqrt(D)
    s = np.exp(s - s.max(axis=1, keepdims=True))
    p = s / s.sum(axis=1, keepdims=True)
    ref = p @ v
    np.testing.assert_allclose(dout.copy_to_host().reshape(N, D), ref, rtol=1e-4, atol=1e-4)
    print("PASS scaled-dot-product-attention")


if __name__ == "__main__":
    test()
