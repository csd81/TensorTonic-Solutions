"""Blokkolt LU-felbontás TRSM-mel, Numba CUDA-val, pivotálás nélkül.

Egy blokk-lépés (k0) menete:
  1. A diagonalis TILE x TILE blokk in-place LU felbontása (shared memory).
  2. L panel: A[i, k0:k0+t] * U_kk = A[i, k0:k0+t]  (jobb oldali TRSM)
  3. U panel: L_kk * A[k0:k0+t, j] = A[k0:k0+t, j]  (bal oldali TRSM)
  4. Trailing GEMM: A[i, j] -= L_ik * U_kj a blokkon kivuli reszre.

A matrix in-place tarolja L-t (also haromszog, egysegdiagonalis) es U-t
(felso haromszog, atloval). Pivotalas nincs; a teszt diagonalisan dominans
matrixot hasznal. Hasznalat: python3 LU-TRSM.py
"""

import time

import numpy as np
from numba import cuda, float32

TILE = 16
TILE_PAD = TILE + 1
TPB_1D = 256


@cuda.jit
def diagonal_factorize_kernel(A, k0, t, n):
    """A[k0:k0+t, k0:k0+t] in-place LU felbontasa shared memoryban."""
    s_A = cuda.shared.array((TILE, TILE_PAD), float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = k0 + ty
    col = k0 + tx

    if row < n and col < n:
        s_A[ty, tx] = A[row, col]
    else:
        s_A[ty, tx] = 0.0

    cuda.syncthreads()

    for k in range(t):
        # A pivot alatti oszlop normalizalasa (szorzok kepzese)
        if ty > k and tx == k:
            s_A[ty, k] /= s_A[k, k]
        cuda.syncthreads()

        # A belso reszmatrix frissitese
        if ty > k and tx > k:
            s_A[ty, tx] -= s_A[ty, k] * s_A[k, tx]
        cuda.syncthreads()

    if row < n and col < n:
        A[row, col] = s_A[ty, tx]


@cuda.jit
def panel_l_trsm_kernel(A, k0, t, n):
    """L_ik = A_ik * U_kk^{-1}: jobb oldali haromszog-megoldas soronkent.

    egy szal egy teljes panel-sort old meg (j = 0 .. t-1):
      X[i,j] = (B[i,j] - sum_{p<j} X[i,p] * U[p,j]) / U[j,j]
    """
    i = cuda.grid(1) + k0 + t
    if i < n:
        for j in range(t):
            s = 0.0
            for p in range(j):
                s += A[i, k0 + p] * A[k0 + p, k0 + j]
            A[i, k0 + j] = (A[i, k0 + j] - s) / A[k0 + j, k0 + j]


@cuda.jit
def panel_u_trsm_kernel(A, k0, t, n):
    """L_kk * U_kj = A_kj: bal oldali haromszog-megoldas oszloponkent.

    egy szal egy teljes panel-oszlopot old meg (i = 0 .. t-1):
      X[i,j] = B[i,j] - sum_{p<i} L[i,p] * X[p,j]
    """
    j = cuda.grid(1) + k0 + t
    if j < n:
        for i in range(t):
            s = 0.0
            for p in range(i):
                s += A[k0 + i, k0 + p] * A[k0 + p, j]
            A[k0 + i, j] -= s


@cuda.jit
def blocked_lu_gemm_kernel(A, k0, n):
    """Trailing GEMM: A[i,j] -= sum_p L[i, k0+p] * U[k0+p, j]."""
    s_L = cuda.shared.array((TILE, TILE_PAD), float32)
    s_U = cuda.shared.array((TILE, TILE_PAD), float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    row = (cuda.blockIdx.y + 1) * TILE + ty + k0
    col = (cuda.blockIdx.x + 1) * TILE + tx + k0

    if row < n and (k0 + tx) < n:
        s_L[ty, tx] = A[row, k0 + tx]
    else:
        s_L[ty, tx] = 0.0

    if (k0 + ty) < n and col < n:
        s_U[ty, tx] = A[k0 + ty, col]
    else:
        s_U[ty, tx] = 0.0

    cuda.syncthreads()

    if row < n and col < n:
        acc = 0.0
        for p in range(TILE):
            acc += s_L[ty, p] * s_U[p, tx]
        A[row, col] -= acc


def blocked_lu_gpu(A):
    """Visszaadja (L, U, stats); A host oldali (n, n) matrix."""
    A32 = np.ascontiguousarray(A, dtype=np.float32)
    n = A32.shape[0]

    h2d_start = time.perf_counter()
    d_A = cuda.to_device(A32)
    cuda.synchronize()
    h2d_ms = (time.perf_counter() - h2d_start) * 1000.0

    start_event = cuda.event()
    end_event = cuda.event()
    start_event.record()

    block_threads = (TILE, TILE)
    for k0 in range(0, n, TILE):
        t = min(TILE, n - k0)

        diagonal_factorize_kernel[(1, 1), block_threads](d_A, k0, t, n)

        if k0 + t < n:
            rem = n - k0 - t

            panel_blocks = (rem + TPB_1D - 1) // TPB_1D
            panel_l_trsm_kernel[panel_blocks, TPB_1D](d_A, k0, t, n)
            panel_u_trsm_kernel[panel_blocks, TPB_1D](d_A, k0, t, n)

            tiles = (rem + TILE - 1) // TILE
            blocked_lu_gemm_kernel[(tiles, tiles), block_threads](d_A, k0, n)

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
    rng = np.random.default_rng(1)
    A = rng.random((n, n)) + n * np.eye(n)  # diagonalisan dominans

    # JIT bemelegites: 32x32-en mar minden kernel lefut
    blocked_lu_gpu(A[:32, :32])

    L, U, stats = blocked_lu_gpu(A)
    rel_err = float(np.linalg.norm(A - L.astype(np.float64) @ U.astype(np.float64))
                    / np.linalg.norm(A))

    from scipy.linalg import lu as scipy_lu

    cpu_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        scipy_lu(A)
        cpu_times.append((time.perf_counter() - t0) * 1000.0)
    cpu_ms = float(np.median(cpu_times))

    print(f"Blokkolt LU (TRSM), n={n}, TILE={TILE}")
    print(f"  GPU H2D:     {stats['h2d_ms']:.2f} ms")
    print(f"  GPU kernel:  {stats['kernel_ms']:.2f} ms")
    print(f"  GPU D2H:     {stats['d2h_ms']:.2f} ms")
    print(f"  GPU total:   {stats['total_ms']:.2f} ms")
    print(f"  CPU SciPy:   {cpu_ms:.2f} ms (median/3)")
    print(f"  Rel. hiba ||A-LU||_F/||A||_F: {rel_err:.3e}")
    print(f"  min |U_ii|: {np.min(np.abs(np.diag(U))):.3e}")
