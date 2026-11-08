"""Numba port of ../cuda-basics/hadamard-product/hadamard-product.cu"""

import numpy as np
from numba import cuda


@cuda.jit
def hadamard_kernel(A, B, C, M, N):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < M and j < N:
        idx = i * N + j
        C[idx] = A[idx] * B[idx]


def solve(A, B, C, M, N):
    threads = (16, 16)
    blocks = ((N + 15) // 16, (M + 15) // 16)
    hadamard_kernel[blocks, threads](A, B, C, M, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    M, N = 37, 53
    A = rng.standard_normal((M, N), dtype=np.float32)
    B = rng.standard_normal((M, N), dtype=np.float32)
    C = np.empty_like(A)
    # The CUDA kernels take flat C-style pointers, so pass 1-D arrays.
    dA = cuda.to_device(A.reshape(-1))
    dB = cuda.to_device(B.reshape(-1))
    dC = cuda.to_device(C.reshape(-1))
    solve(dA, dB, dC, M, N)
    np.testing.assert_allclose(dC.copy_to_host().reshape(M, N), A * B, rtol=1e-6, atol=1e-6)
    print("PASS hadamard-product")


if __name__ == "__main__":
    test()
