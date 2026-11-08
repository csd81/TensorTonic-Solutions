"""2D Poisson-egyenlet optimalizált megoldója: Red-Black SOR.

A fájlnév megtartott, de a naiv Jacobihoz képest az optimalizálás a
README Red-Black / SOR javaslatát követi:

  * in-place frissítés egyetlen rácson (nincs ping-pong puffer),
  * sakktábla (red-black) sorrend: a piros és fekete pontok külön
    kernelben, egymás után frissülnek, így nincs adatverseny,
  * SOR relaxáció omega = 2 / (1 + sin(pi/N)) értékkel,
  * a konvergenciát nem minden iterációban, hanem periodikusan
    (alapból 20 lépésenként) ellenőrzi egy reziduum-max kernel.

Futtatás: python3 2D-Poisson-RedBlack-SOR.py
"""

import math
import time

import numpy as np
from numba import cuda, float32

BLOCK_X = 32
BLOCK_Y = 8


@cuda.jit
def sor_red_kernel(u, f, h2, omega):
    col, row = cuda.grid(2)
    rows, cols = u.shape
    if 0 < row < rows - 1 and 0 < col < cols - 1 and ((row + col) & 1) == 0:
        gs = 0.25 * (
            u[row + 1, col] + u[row - 1, col] +
            u[row, col + 1] + u[row, col - 1] -
            h2 * f[row, col]
        )
        u[row, col] += omega * (gs - u[row, col])


@cuda.jit
def sor_black_kernel(u, f, h2, omega):
    col, row = cuda.grid(2)
    rows, cols = u.shape
    if 0 < row < rows - 1 and 0 < col < cols - 1 and ((row + col) & 1) == 1:
        gs = 0.25 * (
            u[row + 1, col] + u[row - 1, col] +
            u[row, col + 1] + u[row, col - 1] -
            h2 * f[row, col]
        )
        u[row, col] += omega * (gs - u[row, col])


@cuda.jit
def residual_max_kernel(u, f, h2, out):
    col, row = cuda.grid(2)
    rows, cols = u.shape
    if 0 < row < rows - 1 and 0 < col < cols - 1:
        lap = (
            u[row + 1, col] + u[row - 1, col] +
            u[row, col + 1] + u[row, col - 1] -
            4.0 * u[row, col]
        ) / h2
        diff = abs(lap - f[row, col])
        cuda.atomic.max(out, 0, float32(diff))


def run_poisson_solver(N=1024, max_iters=10000, tol=1e-4, check_every=20):
    if N < 3:
        raise ValueError("N must be at least 3")
    if check_every < 1:
        raise ValueError("check_every must be >= 1")

    h = 1.0 / (N - 1)
    h2 = h * h
    omega = 2.0 / (1.0 + math.sin(math.pi / N))

    u_host = np.zeros((N, N), dtype=np.float64)
    f_host = np.ones((N, N), dtype=np.float64)

    h2d_start = time.perf_counter()
    d_u = cuda.to_device(u_host)
    d_f = cuda.to_device(f_host)
    d_res = cuda.device_array(1, dtype=np.float32)
    cuda.synchronize()
    h2d_ms = (time.perf_counter() - h2d_start) * 1000.0

    block = (BLOCK_X, BLOCK_Y)
    grid = ((N + BLOCK_X - 1) // BLOCK_X, (N + BLOCK_Y - 1) // BLOCK_Y)

    start_evt = cuda.event()
    stop_evt = cuda.event()
    start_evt.record()

    converged = False
    residual = float("inf")
    iterations = 0
    for it in range(max_iters):
        sor_red_kernel[grid, block](d_u, d_f, h2, omega)
        sor_black_kernel[grid, block](d_u, d_f, h2, omega)
        iterations = it + 1

        if iterations % check_every == 0 or iterations == max_iters:
            d_res[0] = 0.0
            residual_max_kernel[grid, block](d_u, d_f, h2, d_res)
            cuda.synchronize()
            residual = float(d_res.copy_to_host()[0])
            if residual < tol:
                converged = True
                break

    stop_evt.record()
    stop_evt.synchronize()
    kernel_ms = cuda.event_elapsed_time(start_evt, stop_evt)

    d2h_start = time.perf_counter()
    result = d_u.copy_to_host()
    cuda.synchronize()
    d2h_ms = (time.perf_counter() - d2h_start) * 1000.0

    stats = {
        "h2d_ms": h2d_ms,
        "kernel_ms": kernel_ms,
        "d2h_ms": d2h_ms,
        "total_ms": h2d_ms + kernel_ms + d2h_ms,
        "iterations": iterations,
        "converged": converged,
        "residual": residual,
        "omega": float(omega),
    }
    return result, stats


def sparse_reference(N):
    """Közvetlen ritka lineáris megoldás a diszkrét Poisson-rendszerre."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    n = N - 2
    h = 1.0 / (N - 1)
    h2 = h * h
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    I = sp.eye(n)
    A = (sp.kron(I, T) + sp.kron(T, I)).tocsr()
    rhs = -h2 * np.ones(n * n)
    interior = spla.spsolve(A, rhs)

    u = np.zeros((N, N), dtype=np.float64)
    u[1:-1, 1:-1] = interior.reshape(n, n)
    return u


def max_residual_host(u, N):
    h = 1.0 / (N - 1)
    h2 = h * h
    lap = (
        u[2:, 1:-1] + u[:-2, 1:-1] +
        u[1:-1, 2:] + u[1:-1, :-2] -
        4.0 * u[1:-1, 1:-1]
    ) / h2
    return float(np.max(np.abs(lap - 1.0)))


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    # JIT bemelegites
    run_poisson_solver(N=64, max_iters=5)

    # Numerikus ellenorzes a ritka direkt megoldassal
    N_CHECK = 256
    gpu_u, check_stats = run_poisson_solver(
        N_CHECK, max_iters=10000, tol=1e-5, check_every=20)
    ref_u = sparse_reference(N_CHECK)
    max_err = float(np.max(np.abs(gpu_u - ref_u)))
    print(f"Ellenorzes N={N_CHECK}: iterations={check_stats['iterations']}, "
          f"residual={check_stats['residual']:.3e}, "
          f"max |GPU-SciPy|={max_err:.3e}")

    # Benchmark
    print("\nRed-Black SOR benchmark (omega optimalis)")
    print(f"{'N':>6} {'omega':>8} {'iter':>7} {'residual':>12} "
          f"{'kernel':>10} {'total':>10} {'conv':>6}")
    for N in (256, 512, 1024):
        u, stats = run_poisson_solver(
            N, max_iters=20000, tol=1e-4, check_every=20)
        print(f"{N:>6} {stats['omega']:>8.5f} {stats['iterations']:>7} "
              f"{stats['residual']:>12.3e} {stats['kernel_ms']:>9.2f}ms "
              f"{stats['total_ms']:>9.2f}ms {str(stats['converged']):>6}")

    # Osszehasonlitas a naiv Jacobi valtozattal ugyanazon a racsmereten.
    try:
        import importlib.util
        import pathlib

        naive_path = pathlib.Path(__file__).with_name(
            "2D-Poisson-Jacobi.py")
        spec = importlib.util.spec_from_file_location("poisson_naive", naive_path)
        naive = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(naive)
        naive.run_poisson_solver(N=64, max_iters=5)  # warm-up

        print("\nNaiv Jacobi osszehasonlitas (N=256)")
        for iters in (500, 5000):
            naive_u, naive_stats = naive.run_poisson_solver(256, iters)
            res = max_residual_host(naive_u, 256)
            print(f"  {iters:>5} iter: residual={res:.3e}, "
                  f"kernel={naive_stats['kernel_ms']:.1f} ms")
    except Exception as exc:  # pragma: no cover
        print(f"Naiv osszehasonlitas kihagyva: {exc}")
