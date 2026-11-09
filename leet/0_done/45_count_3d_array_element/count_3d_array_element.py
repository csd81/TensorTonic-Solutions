"""Numba CUDA port of count_3d_array_element.cu.

Count elements equal to P in an N x M x K int array.

At the benchmark (500^3 = 125,000,000 elements) the kernel reads 500 MB, so
it is read-bound: a C++ int4 read measures 5.78 ms (86.5 GB/s) and both the
scalar and vectorized C++ kernels are at that floor.  This port keeps the
scalar grid-stride mapping with a per-thread count and one atomicAdd per
thread that found matches (Numba has no int4 type).

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, int32, int64

THREADS = 256


@cuda.jit(cache=True)
def zero_kernel(a):
    if cuda.grid(1) == 0:
        a[0] = 0


@cuda.jit(cache=True)
def count_kernel(input, output, total, P):
    # int64 index: total can exceed 2^31/... (up to 1e9 here, which fits
    # int32, but the stride arithmetic stays 64-bit for safety).
    idx = int64(cuda.blockIdx.x) * THREADS + int64(cuda.threadIdx.x)
    stride = int64(cuda.blockDim.x) * int64(cuda.gridDim.x)
    local_count = int32(0)
    for i in range(idx, total, stride):
        if input[i] == P:
            local_count += 1
    if local_count > 0:
        cuda.atomic.add(output, 0, local_count)


def solve(input, output, N, M, K, P):
    """input is an int32 device array of N*M*K elements; output is a length-1
    int32 device array."""
    if N <= 0 or M <= 0 or K <= 0:
        return
    if input is None or output is None:
        return
    total = N * M * K
    blocks = (total + THREADS - 1) // THREADS
    if blocks > 1024:
        blocks = 1024
    zero_kernel[1, 1](output)
    count_kernel[blocks, THREADS](input, output, int64(total), int32(P))
    cuda.synchronize()


def _run(name, a, P):
    total = int(a.size)
    ref = int(np.count_nonzero(a == P))
    d_in = cuda.to_device(np.ascontiguousarray(a, dtype=np.int32).reshape(-1))
    d_out = cuda.device_array(1, dtype=np.int32)
    solve(d_in, d_out, int(a.shape[0]), int(a.shape[1]), int(a.shape[2]), P)
    got = int(d_out.copy_to_host()[0])
    ok = got == ref
    print(f"{name:>18} N={a.shape[0]} M={a.shape[1]} K={a.shape[2]} P={P}: "
          f"got={got} ref={ref} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([1, 2, 3, 4, 5, 1, 1, 1, 1, 2, 2, 2],
                                     np.int32).reshape(2, 2, 3), 1)
    ok &= _run("example 2", np.array([5, 10, 5, 2, 2, 2], np.int32)
               .reshape(1, 3, 2), 1)
    for shape, P in (((1, 1, 1), 1), ((1, 1, 1), 50), ((10, 20, 30), 7),
                     ((100, 100, 100), 100), ((999, 1, 1), 42),
                     ((1, 1000, 61), 50)):
        a = rng.integers(1, 101, shape, dtype=np.int32)
        ok &= _run(f"random {shape[0]}x{shape[1]}x{shape[2]}", a, P)

    # Benchmark: 500 x 500 x 500.
    import statistics
    import time

    shape = (500, 500, 500)
    P = 50
    a = rng.integers(1, 101, shape, dtype=np.int32)
    d_in = cuda.to_device(a.reshape(-1))
    d_out = cuda.device_array(1, dtype=np.int32)
    solve(d_in, d_out, 500, 500, 500, P)  # warm-up + JIT compile
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        solve(d_in, d_out, 500, 500, 500, P)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    got = int(d_out.copy_to_host()[0])
    ref = int(np.count_nonzero(a == P))
    bytes_read = a.size * 4
    print(f"Perf 500x500x500 P={P}: {ms:.4f} ms, "
          f"{bytes_read / (ms * 1e6):.1f} GB/s, got={got} ref={ref}, "
          f"{'OK' if got == ref else 'FAIL'}")
    ok &= got == ref
    print("ALL OK" if ok else "FAILURES")
