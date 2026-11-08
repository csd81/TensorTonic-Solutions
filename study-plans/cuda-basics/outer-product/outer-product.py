"""Numba port of ../cuda-basics/outer-product/outer-product.cu"""

import numpy as np
from numba import cuda, float32


@cuda.jit
def outer_product_kernel(a, b, C, M, N):
    s_a = cuda.shared.array(16, float32)
    s_b = cuda.shared.array(16, float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    j = cuda.blockIdx.x * 16 + tx
    i = cuda.blockIdx.y * 16 + ty

    if tx == 0:
        s_a[ty] = a[i] if i < M else 0.0
    if ty == 0:
        s_b[tx] = b[j] if j < N else 0.0

    cuda.syncthreads()

    if i < M and j < N:
        C[i * N + j] = s_a[ty] * s_b[tx]


def solve(a, b, C, M, N):
    threads = (16, 16)
    blocks = ((N + 15) // 16, (M + 15) // 16)
    outer_product_kernel[blocks, threads](a, b, C, M, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    M, N = 37, 53
    a = rng.standard_normal(M, dtype=np.float32)
    b = rng.standard_normal(N, dtype=np.float32)
    C = np.empty((M, N), dtype=np.float32)
    da, db = cuda.to_device(a), cuda.to_device(b)
    dC = cuda.to_device(C.reshape(-1))
    solve(da, db, dC, M, N)
    ref = np.outer(a.astype(np.float64), b.astype(np.float64))
    np.testing.assert_allclose(dC.copy_to_host().reshape(M, N), ref, rtol=1e-6, atol=1e-6)
    print("PASS outer-product")


if __name__ == "__main__":
    test()
