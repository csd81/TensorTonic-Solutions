"""Numba CUDA port of count_2d_array_element.cu.

Count elements equal to P in an N x M int32 array.

At the benchmark (10,000 x 10,000 = 100,000,000 elements) the kernel reads
400 MB, so it is read-bound: the C++ scalar and int4 kernels both measure
4.60-4.68 ms (86 GB/s), at the read ceiling.  This port keeps the scalar
grid-stride mapping with a per-thread count and one atomicAdd per thread that
found matches (Numba has no int4 type).

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
    idx = int64(cuda.blockIdx.x) * THREADS + int64(cuda.threadIdx.x)
    stride = int64(cuda.blockDim.x) * int64(cuda.gridDim.x)
    local = int32(0)
    for i in range(idx, total, stride):
        if input[i] == P:
            local += 1
    if local > 0:
        cuda.atomic.add(output, 0, local)


def solve(inp, output, N, M, P):
    """inp is an int32 device array of N*M elements; output is a length-1
    int32 device array."""
    if inp is None or output is None:
        return
    total = N * M
    zero_kernel[1, 1](output)
    if total <= 0:
        cuda.synchronize()
        return
    blocks = (total + THREADS - 1) // THREADS
    if blocks > 4096:
        blocks = 4096
    count_kernel[blocks, THREADS](inp, output, int64(total), int32(P))
    cuda.synchronize()


def _run(name, a, P):
    ref = int(np.count_nonzero(a == P))
    d_in = cuda.to_device(np.ascontiguousarray(a, dtype=np.int32).reshape(-1))
    d_out = cuda.device_array(1, dtype=np.int32)
    solve(d_in, d_out, int(a.shape[0]), int(a.shape[1]), P)
    got = int(d_out.copy_to_host()[0])
    ok = got == ref
    print(f"{name:>18} N={a.shape[0]} M={a.shape[1]} P={P}: got={got} "
          f"ref={ref} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([[1, 2, 3], [4, 5, 1]], np.int32), 1)
    ok &= _run("example 2", np.array([[5, 10], [5, 2]], np.int32), 1)
    for shape, P in (((1, 1), 1), ((1, 1), 50), ((10, 20), 7),
                     ((100, 100), 100), ((1, 1000000), 42), ((999, 1001), 50)):
        a = rng.integers(1, 101, shape, dtype=np.int32)
        ok &= _run(f"random {shape[0]}x{shape[1]}", a, P)

    # Benchmark: 10,000 x 10,000.
    import statistics
    import time

    N = M = 10000
    P = 1
    a = rng.integers(1, 101, (N, M), dtype=np.int32)
    d_in = cuda.to_device(a.reshape(-1))
    d_out = cuda.device_array(1, dtype=np.int32)
    solve(d_in, d_out, N, M, P)  # warm-up, also triggers the JIT compile
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        solve(d_in, d_out, N, M, P)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    got = int(d_out.copy_to_host()[0])
    ref = int(np.count_nonzero(a == P))
    print(f"Perf {N}x{M} P={P}: {ms:.4f} ms, "
          f"{N * M * 4 / (ms * 1e6):.1f} GB/s, got={got} ref={ref}, "
          f"{'OK' if got == ref else 'FAIL'}")
    ok &= got == ref
    print("ALL OK" if ok else "FAILURES")
