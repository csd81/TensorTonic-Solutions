"""Numba CUDA port of jacobi_stencil_2d.cu.

5-point Jacobi stencil: interior cells become the average of their four
cardinal neighbours; boundary cells are copied through.

At the benchmark (8192 x 8192) the kernel moves 537 MB, so it is
memory-bound: a plain float4 copy measures ~7.0 ms (76 GB/s).  In C++ the
starter's scalar kernel measures 7.2-7.6 ms and the shipped 4-cells-per-thread
float4 kernel measures 6.7-7.0 ms (it loads the top, bottom and centre rows
as float4 and forms left/right neighbours in registers).  This port keeps the
scalar one-cell-per-thread mapping, since Numba has no float4 type.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32

TX, TY = 32, 8


@cuda.jit(cache=True)
def jacobi_kernel(input, output, rows, cols):
    r = cuda.blockIdx.y * TY + cuda.threadIdx.y
    c = cuda.blockIdx.x * TX + cuda.threadIdx.x
    if r < rows and c < cols:
        i = r * cols + c
        if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
            output[i] = input[i]
        else:
            output[i] = float32(0.25) * (
                input[i - cols] + input[i + cols] + input[i - 1] + input[i + 1])


def solve(input, output, rows, cols):
    """input and output are float32 device arrays of length rows*cols."""
    if rows <= 0 or cols <= 0:
        return
    if input is None or output is None:
        return
    grid = ((cols + TX - 1) // TX, (rows + TY - 1) // TY)
    jacobi_kernel[grid, (TX, TY)](input, output, rows, cols)
    cuda.synchronize()


def _ref(a, rows, cols):
    a64 = a.astype(np.float64).reshape(rows, cols)
    out = a64.copy()
    out[1:-1, 1:-1] = 0.25 * (a64[:-2, 1:-1] + a64[2:, 1:-1]
                              + a64[1:-1, :-2] + a64[1:-1, 2:])
    return out.reshape(-1)


def _run(name, a, rows, cols):
    ref = _ref(a, rows, cols)
    d_in = cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    d_out = cuda.device_array(a.size, dtype=np.float32)
    solve(d_in, d_out, rows, cols)
    got = d_out.copy_to_host().astype(np.float64)
    abs_err = np.abs(got - ref)
    bad = int(np.count_nonzero(
        abs_err > 1e-4 * np.maximum(1.0, np.abs(ref))))
    ok = bad == 0
    print(f"{name:>18} {rows}x{cols}: max_abs={np.max(abs_err):.2e} "
          f"bad={bad} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example", np.arange(1, 17, dtype=np.float32), 4, 4)
    for r, c in ((1, 1), (2, 2), (3, 5), (5, 3), (17, 33), (64, 64), (100, 37)):
        a = rng.uniform(-100.0, 100.0, r * c).astype(np.float32)
        ok &= _run(f"random {r}x{c}", a, r, c)

    # Benchmark: 8192 x 8192.
    import statistics
    import time

    R = C = 8192
    a = rng.uniform(-100.0, 100.0, R * C).astype(np.float32)
    d_in = cuda.to_device(a)
    d_out = cuda.device_array(R * C, dtype=np.float32)
    solve(d_in, d_out, R, C)  # warm-up, also triggers the JIT compile
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(d_in, d_out, R, C)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    ref = _ref(a, R, C)
    got = d_out.copy_to_host().astype(np.float64)
    bad = int(np.count_nonzero(
        np.abs(got - ref) > 1e-4 * np.maximum(1.0, np.abs(ref))))
    print(f"Perf {R}x{C}: {ms:.4f} ms, {2 * R * C * 4 / (ms * 1e6):.1f} GB/s, "
          f"bad={bad}, {'OK' if bad == 0 else 'FAIL'}")
    ok &= bad == 0
    print("ALL OK" if ok else "FAILURES")
