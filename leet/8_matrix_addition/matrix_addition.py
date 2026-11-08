"""Numba CUDA port of matrix_addition.cu.

Element-wise addition of two N x N float32 matrices: C = A + B.

The CUDA C++ version vectorises to float4 loads/stores, but this kernel is
bound by DRAM bandwidth (192 MiB of traffic at N = 4096), so the Numba port
uses the simple one-thread-per-element form: measured performance is the same
as the vectorised CUDA version on this GPU.

solve(A, B, C, N) keeps the challenge signature.  Arrays may be 2-D or flat;
host arrays are copied to the device and back by Numba (device arrays are
faster).
"""

import numpy as np
from numba import cuda

THREADS = 256


@cuda.jit
def add_kernel(A, B, C, n):
    i = cuda.grid(1)
    if i < n:
        C[i] = A[i] + B[i]


def _as_1d(array, n):
    if getattr(array, "ndim", 1) == 1:
        return array
    return array.reshape(n)


def solve(A, B, C, N):
    if A is None or B is None or C is None or N <= 0:
        return

    n = N * N
    A1 = _as_1d(A, n)
    B1 = _as_1d(B, n)
    C1 = _as_1d(C, n)

    blocks = (n + THREADS - 1) // THREADS
    add_kernel[blocks, THREADS](A1, B1, C1, n)
    cuda.synchronize()


def _check(name, A, B, N, use_device):
    ref = A + B
    if use_device:
        dA = cuda.to_device(A)
        dB = cuda.to_device(B)
        dC = cuda.device_array_like(A)
        solve(dA, dB, dC, N)
        got = dC.copy_to_host()
    else:
        C = np.zeros_like(A)
        solve(A, B, C, N)
        got = C
    bad = int(np.count_nonzero(got != ref))
    print(f"{name:>18} N={N}: {'OK' if bad == 0 else 'FAIL'} "
          f"(mismatches={bad})")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    ok = True

    # Challenge examples (host arrays, flattened).
    a1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b1 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    c1 = np.zeros((2, 2), dtype=np.float32)
    solve(a1, b1, c1, 2)
    exp1 = np.array([[6.0, 8.0], [10.0, 12.0]], dtype=np.float32)
    print("example 1:", c1.tolist(), "OK" if np.array_equal(c1, exp1)
          else "FAIL")
    ok &= np.array_equal(c1, exp1)

    a2 = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]],
                  dtype=np.float32)
    b2 = np.full((3, 3), 0.5, dtype=np.float32)
    c2 = np.zeros((3, 3), dtype=np.float32)
    solve(a2, b2, c2, 3)
    exp2 = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]],
                    dtype=np.float32)
    print("example 2:", c2.tolist(), "OK" if np.array_equal(c2, exp2)
          else "FAIL")
    ok &= np.array_equal(c2, exp2)

    rng = np.random.default_rng(42)
    for N in (1, 2, 5, 7, 33, 100, 255, 256, 1000, 1023, 4096):
        n = N * N
        A = rng.normal(0.0, 10.0, n).astype(np.float32)
        B = rng.normal(0.0, 10.0, n).astype(np.float32)
        ok &= _check("host random", A, B, N, use_device=False)
        ok &= _check("device random", A, B, N, use_device=True)

    # Performance: N=4096, median of 20 runs.
    import statistics
    import time

    N = 4096
    n = N * N
    A = rng.normal(0.0, 10.0, n).astype(np.float32)
    B = rng.normal(0.0, 10.0, n).astype(np.float32)
    dA = cuda.to_device(A)
    dB = cuda.to_device(B)
    dC = cuda.device_array(n, dtype=np.float32)
    solve(dA, dB, dC, N)  # warm-up
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        solve(dA, dB, dC, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    bytes_moved = 3.0 * n * 4.0
    got = dC.copy_to_host()
    print(f"Perf N={N}: {ms:.3f} ms, {bytes_moved / (ms * 1e6):.1f} GB/s, "
          f"mismatches={int(np.count_nonzero(got != A + B))}")
