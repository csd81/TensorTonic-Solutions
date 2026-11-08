"""2D Poisson-egyenlet megoldása Jacobi-iterációval, naiv globális memóriás
Numba CUDA kernellel.

A rács (N x N) belső pontjaira:
    u_new[i,j] = 0.25 * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]
                         - h^2 * f[i,j])
Dirichlet-perem: a szélső sorok/oszlopok rögzítettek (nullák).
Ping-pong pufferek, a peremeket a kernel nem írja.

Futtatás: python3 2D-Poisson-Jacobi.py
"""

import time

import numpy as np
from numba import cuda

TPB_X = 16
TPB_Y = 16


@cuda.jit
def poisson_jacobi_naive_kernel(u_old, u_new, f, h2):
    # cuda.grid(2) -> (x, y) = (oszlop, sor)
    col, row = cuda.grid(2)
    rows, cols = u_old.shape

    if 0 < row < rows - 1 and 0 < col < cols - 1:
        u_new[row, col] = 0.25 * (
            u_old[row + 1, col] + u_old[row - 1, col] +
            u_old[row, col + 1] + u_old[row, col - 1] -
            h2 * f[row, col]
        )


def run_poisson_solver(N=1024, max_iters=1000):
    """Visszaadja (u, stats); u az utolso iteracio eredmenye."""
    if N < 3:
        raise ValueError("N must be at least 3")

    h = 1.0 / (N - 1)
    h2 = h * h

    u_host = np.zeros((N, N), dtype=np.float32)
    f_host = np.ones((N, N), dtype=np.float32)

    h2d_start = time.perf_counter()
    d_old = cuda.to_device(u_host)
    d_new = cuda.to_device(u_host)
    d_f = cuda.to_device(f_host)
    cuda.synchronize()
    h2d_ms = (time.perf_counter() - h2d_start) * 1000.0

    block = (TPB_X, TPB_Y)
    grid = ((N + TPB_X - 1) // TPB_X, (N + TPB_Y - 1) // TPB_Y)

    start_evt = cuda.event()
    stop_evt = cuda.event()
    start_evt.record()

    for _ in range(max_iters):
        poisson_jacobi_naive_kernel[grid, block](d_old, d_new, d_f, h2)
        # Pointer swap: a frissen szamitott eredmeny kerul az u_old-ba.
        d_old, d_new = d_new, d_old

    stop_evt.record()
    stop_evt.synchronize()
    kernel_ms = cuda.event_elapsed_time(start_evt, stop_evt)

    d2h_start = time.perf_counter()
    result = d_old.copy_to_host()
    cuda.synchronize()
    d2h_ms = (time.perf_counter() - d2h_start) * 1000.0

    stats = {
        "h2d_ms": h2d_ms,
        "kernel_ms": kernel_ms,
        "d2h_ms": d2h_ms,
        "total_ms": h2d_ms + kernel_ms + d2h_ms,
        "iterations": max_iters,
    }
    return result, stats


def cpu_jacobi_reference(N, iters):
    """Vektorizalt NumPy referencia ugyanarra az iteracioszamra."""
    h = 1.0 / (N - 1)
    h2 = h * h
    u = np.zeros((N, N), dtype=np.float64)
    f = np.ones((N, N), dtype=np.float64)
    for _ in range(iters):
        u_new = u.copy()
        u_new[1:-1, 1:-1] = 0.25 * (
            u[2:, 1:-1] + u[:-2, 1:-1] +
            u[1:-1, 2:] + u[1:-1, :-2] -
            h2 * f[1:-1, 1:-1]
        )
        u = u_new
    return u


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    # JIT bemelegites
    run_poisson_solver(N=64, max_iters=5)

    # Numerikus ellenorzes kisebb racsban, azonos iteracioszammal
    N_CHECK, ITERS_CHECK = 256, 300
    gpu_u, _ = run_poisson_solver(N_CHECK, ITERS_CHECK)
    cpu_u = cpu_jacobi_reference(N_CHECK, ITERS_CHECK)
    max_err = float(np.max(np.abs(gpu_u - cpu_u)))
    print(f"Ellenorzes N={N_CHECK}, iters={ITERS_CHECK}: "
          f"max |GPU-CPU| = {max_err:.3e}")

    # Benchmark
    print("\nNaiv Jacobi benchmark")
    print(f"{'N':>6} {'iter':>6} {'H2D':>9} {'kernel':>10} {'D2H':>8} {'total':>9}")
    for N, iters in ((256, 500), (512, 500), (1024, 500)):
        _, stats = run_poisson_solver(N, iters)
        print(f"{N:>6} {iters:>6} {stats['h2d_ms']:>8.2f}ms "
              f"{stats['kernel_ms']:>9.2f}ms {stats['d2h_ms']:>7.2f}ms "
              f"{stats['total_ms']:>8.2f}ms")
