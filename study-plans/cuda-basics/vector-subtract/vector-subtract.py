"""Numba port of ../cuda-basics/vector-subtract/vector-subtract.cu"""

import numpy as np
from numba import cuda


@cuda.jit
def vector_sub(A, B, C, N):
    i = cuda.grid(1)
    if i < N:
        C[i] = A[i] - B[i]


def solve(A, B, C, N):
    threads = 256
    blocks = (N + threads - 1) // threads
    vector_sub[blocks, threads](A, B, C, N)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N = 10_000
    A = rng.standard_normal(N, dtype=np.float32)
    B = rng.standard_normal(N, dtype=np.float32)
    C = np.empty_like(A)
    dA, dB, dC = cuda.to_device(A), cuda.to_device(B), cuda.to_device(C)
    solve(dA, dB, dC, N)
    np.testing.assert_allclose(dC.copy_to_host(), A - B, rtol=1e-6, atol=1e-6)
    print("PASS vector-subtract")


if __name__ == "__main__":
    test()
