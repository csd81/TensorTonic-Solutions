"""Numba CUDA port of vector_add.cu.

C[i] = A[i] + B[i], float32, flat 1-D vectors (or reshaped from N elements).
solve(A, B, C, N) keeps the challenge signature and writes into C.
Host NumPy arrays and Numba device arrays are both accepted.
"""

import numpy as np
from numba import cuda


@cuda.jit
def vector_add(A, B, C, N):
    i = cuda.grid(1)
    if i < N:
        C[i] = A[i] + B[i]


def _as_1d(array, n):
    if getattr(array, "ndim", 1) == 1:
        return array
    return array.reshape(n)


def solve(A, B, C, N):
    if N < 1:
        raise ValueError("N must be positive")

    A1 = _as_1d(A, N)
    B1 = _as_1d(B, N)
    C1 = _as_1d(C, N)

    threads = 256
    blocks = (N + threads - 1) // threads
    vector_add[blocks, threads](A1, B1, C1, N)
    cuda.synchronize()


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(0)
    for N in (4, 1, 1000, 1000003):
        A = rng.standard_normal(N).astype(np.float32)
        B = rng.standard_normal(N).astype(np.float32)
        C = np.zeros(N, dtype=np.float32)

        dA = cuda.to_device(A)
        dB = cuda.to_device(B)
        dC = cuda.to_device(C)
        solve(dA, dB, dC, N)
        got = dC.copy_to_host()

        ref = A.astype(np.float64) + B.astype(np.float64)
        max_err = float(np.max(np.abs(got - ref)))
        rel = max_err / max(1.0, float(np.max(np.abs(ref))))
        print(f"N={N}: max_err={max_err:.3e} rel={rel:.3e} "
              f"{'OK' if rel < 1e-6 else 'FAIL'}")
