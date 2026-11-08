"""Naiv (blokkolatlan) LU-felbontás Numba CUDA-val, pivotálás nélkül.

A matrix in-place tárolja L-t (szigorúan alsó háromszög, egységdiagonális)
és U-t (felső háromszög, átlóval). Minden k lépésben:
  1. A[k+1:, k] /= A[k, k]        (szorzók)
  2. A[k+1:, k+1:] -= A[k+1:, k] * A[k, k+1:]
A külső ciklus host oldalon fut, a két részfeladat külön kernelen.

Pivotálás nincs; a teszt diagonálisan domináns mátrixot használ, ahol ez
numerikusan biztonságos. Használat: python3 LU-naive.py
"""

import time

import numpy as np
from numba import cuda, float32

TPB_1D = 256
BLOCK_2D = (16, 16)


@cuda.jit
def lu_scale_kernel(A, k, n):
    """A[i, k] /= A[k, k] az i > k sorokra."""
    i = cuda.grid(1) + k + 1
    if i < n:
        A[i, k] /= A[k, k]


@cuda.jit
def lu_update_kernel(A, k, n):
    """A[i, j] -= A[i, k] * A[k, j] az i, j > k elemekre."""
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + k + 1
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y + k + 1
    if row < n and col < n:
        A[row, col] -= A[row, k] * A[k, col]


def lu_naive_gpu(A):
    """Visszaadja (L, U, stats); A host oldali (n, n) mátrix."""
    A32 = np.ascontiguousarray(A, dtype=np.float32)
    n = A32.shape[0]

    h2d_start = time.perf_counter()
    d_A = cuda.to_device(A32)
    cuda.synchronize()
    h2d_ms = (time.perf_counter() - h2d_start) * 1000.0

    start_event = cuda.event()
    end_event = cuda.event()
    start_event.record()

    for k in range(n):
        remaining = n - k - 1
        if remaining <= 0:
            continue

        scale_blocks = (remaining + TPB_1D - 1) // TPB_1D
        lu_scale_kernel[scale_blocks, TPB_1D](d_A, k, n)

        grid = (
            (remaining + BLOCK_2D[0] - 1) // BLOCK_2D[0],
            (remaining + BLOCK_2D[1] - 1) // BLOCK_2D[1],
        )
        lu_update_kernel[grid, BLOCK_2D](d_A, k, n)

    end_event.record()
    end_event.synchronize()
    kernel_ms = start_event.elapsed_time(end_event)

    d2h_start = time.perf_counter()
    A_out = d_A.copy_to_host()
    cuda.synchronize()
    d2h_ms = (time.perf_counter() - d2h_start) * 1000.0

    L = np.tril(A_out, -1) + np.eye(n, dtype=np.float32)
    U = np.triu(A_out)

    stats = {
        "h2d_ms": h2d_ms,
        "kernel_ms": kernel_ms,
        "d2h_ms": d2h_ms,
        "total_ms": h2d_ms + kernel_ms + d2h_ms,
    }
    return L, U, stats


if __name__ == "__main__":
    n = 256
    rng = np.random.default_rng(0)
    A = rng.random((n, n)) + n * np.eye(n)  # diagonálisan domináns

    # JIT bemelegítés ugyanilyen típusú, kisebb mátrixon
    lu_naive_gpu(A[:16, :16])

    L, U, stats = lu_naive_gpu(A)
    rel_err = float(np.linalg.norm(A - L.astype(np.float64) @ U.astype(np.float64))
                    / np.linalg.norm(A))

    from scipy.linalg import lu as scipy_lu

    cpu_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        scipy_lu(A)
        cpu_times.append((time.perf_counter() - t0) * 1000.0)
    cpu_ms = float(np.median(cpu_times))

    print(f"Naiv LU, n={n}")
    print(f"  GPU H2D:     {stats['h2d_ms']:.2f} ms")
    print(f"  GPU kernel:  {stats['kernel_ms']:.2f} ms")
    print(f"  GPU D2H:     {stats['d2h_ms']:.2f} ms")
    print(f"  GPU total:   {stats['total_ms']:.2f} ms")
    print(f"  CPU SciPy:   {cpu_ms:.2f} ms (median/3)")
    print(f"  Rel. hiba ||A-LU||_F/||A||_F: {rel_err:.3e}")
    print(f"  min |U_ii|: {np.min(np.abs(np.diag(U))):.3e}")
