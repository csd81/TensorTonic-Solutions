"""Numba CUDA port of weight_dequantization.cu.

Y[i][j] = X[i][j] * S[i/T][j/T] — per-tile scale dequantization.

At the benchmark (8192 x 8192) the kernel moves 537 MB and S is a cached
64x64 table, so it is memory-bound: in C++ the starter's scalar kernel
measures 7.3 ms (73 GB/s) against a 6.6 ms copy floor, and the shipped
float4/4-cells-per-thread kernel measures 6.5-7.0 ms (77-82 GB/s).  This port
keeps the scalar mapping (Numba has no float4 type).

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, int32

TX, TY = 32, 8


@cuda.jit(cache=True)
def dequant_kernel_shift(X, S, Y, M, N, Tshift, S_cols):
    # Every allowed tile size is a power of two, so the tile index is a shift
    # rather than a division.  Numba's integer division is far more expensive
    # than nvcc's (two divisions per element took the port from 7.3 to 12.5 ms,
    # and to 14.1 ms before the indices were forced to int32).  int32 casts:
    # blockIdx/threadIdx promote to 64-bit in Numba otherwise.
    j = int32(cuda.blockIdx.x) * int32(TX) + int32(cuda.threadIdx.x)
    i = int32(cuda.blockIdx.y) * int32(TY) + int32(cuda.threadIdx.y)
    if i < M and j < N:
        idx = i * N + j
        Y[idx] = X[idx] * S[(i >> Tshift) * S_cols + (j >> Tshift)]


@cuda.jit(cache=True)
def dequant_kernel_div(X, S, Y, M, N, T, S_cols):
    # Fallback for a tile size that is not a power of two.
    j = int32(cuda.blockIdx.x) * int32(TX) + int32(cuda.threadIdx.x)
    i = int32(cuda.blockIdx.y) * int32(TY) + int32(cuda.threadIdx.y)
    if i < M and j < N:
        idx = i * N + j
        Y[idx] = X[idx] * S[(i // T) * S_cols + j // T]


def solve(X, S, Y, M, N, TILE_SIZE):
    """X, Y: float32 device arrays M*N; S: float32 device array of
    ceil(M/T)*ceil(N/T)."""
    if M <= 0 or N <= 0 or TILE_SIZE <= 0:
        return
    if X is None or S is None or Y is None:
        return
    s_cols = (N + TILE_SIZE - 1) // TILE_SIZE
    grid = ((N + TX - 1) // TX, (M + TY - 1) // TY)
    if TILE_SIZE & (TILE_SIZE - 1) == 0:
        tshift = int32(TILE_SIZE.bit_length() - 1)
        dequant_kernel_shift[grid, (TX, TY)](X, S, Y, int32(M), int32(N),
                                             tshift, int32(s_cols))
    else:
        dequant_kernel_div[grid, (TX, TY)](X, S, Y, int32(M), int32(N),
                                           int32(TILE_SIZE), int32(s_cols))
    cuda.synchronize()


def _ref(x, s, M, N, T):
    x64 = x.astype(np.float64).reshape(M, N)
    sc = (N + T - 1) // T
    sr = (M + T - 1) // T
    s64 = s.astype(np.float64).reshape(sr, sc)
    # Expand the scale table to M x N.
    scale = np.repeat(np.repeat(s64, T, axis=0), T, axis=1)[:M, :N]
    return (x64 * scale).reshape(-1)


def _run(name, x, s, M, N, T):
    ref = _ref(x, s, M, N, T)
    dx = cuda.to_device(np.ascontiguousarray(x, dtype=np.float32))
    ds = cuda.to_device(np.ascontiguousarray(s, dtype=np.float32))
    dy = cuda.device_array(M * N, dtype=np.float32)
    solve(dx, ds, dy, M, N, T)
    got = dy.copy_to_host().astype(np.float64)
    abs_err = np.abs(got - ref)
    bad = int(np.count_nonzero(abs_err > 1e-4 * np.maximum(1.0, np.abs(ref))))
    ok = bad == 0
    print(f"{name:>16} M={M} N={N} T={T}: max_abs={np.max(abs_err):.2e} "
          f"bad={bad} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example", np.array([10, 10, 5, 5, 10, 10, 5, 5,
                                    2, 2, 8, 8, 2, 2, 8, 8], np.float32),
               np.array([0.5, 2.0, 4.0, 0.25], np.float32), 4, 4, 2)
    for T in (16, 32, 64, 128):
        for M, N in ((1, 1), (16, 16), (17, 33), (100, 37), (1000, 1000)):
            x = rng.uniform(-10.0, 10.0, M * N).astype(np.float32)
            sr = (M + T - 1) // T
            sc = (N + T - 1) // T
            s = rng.uniform(0.01, 4.0, sr * sc).astype(np.float32)
            ok &= _run(f"{M}x{N}", x, s, M, N, T)

    # Benchmark: 8192 x 8192, T = 128.
    import statistics
    import time

    M = N = 8192
    T = 128
    x = rng.uniform(-10.0, 10.0, M * N).astype(np.float32)
    sr = (M + T - 1) // T
    sc = (N + T - 1) // T
    s = rng.uniform(0.01, 4.0, sr * sc).astype(np.float32)
    dx = cuda.to_device(x)
    ds = cuda.to_device(s)
    dy = cuda.device_array(M * N, dtype=np.float32)
    solve(dx, ds, dy, M, N, T)  # warm-up, also triggers the JIT compile
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(dx, ds, dy, M, N, T)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    ref = _ref(x, s, M, N, T)
    got = dy.copy_to_host().astype(np.float64)
    bad = int(np.count_nonzero(np.abs(got - ref) > 1e-4 * np.maximum(1.0, np.abs(ref))))
    print(f"Perf {M}x{N} T={T}: {ms:.4f} ms, "
          f"{2 * M * N * 4 / (ms * 1e6):.1f} GB/s, bad={bad}, "
          f"{'OK' if bad == 0 else 'FAIL'}")
    ok &= bad == 0
    print("ALL OK" if ok else "FAILURES")
