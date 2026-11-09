"""Numba CUDA port of subarray_sum.cu.

Sum of input[S..E] (inclusive), int32.

At the benchmark (N = 100,000,000) the kernel reads 400 MB, so it is
read-bound: a pure int4 read measures ~4.65 ms (86 GB/s), the C++ scalar
naive form 4.95-5.02 ms, and the shipped int4 kernel 4.61-4.63 ms.  This port
keeps the scalar grid-stride mapping (Numba has no int4 type).

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
def sum_kernel(input, output, S, E):
    idx = int64(cuda.blockIdx.x) * THREADS + int64(cuda.threadIdx.x)
    stride = int64(cuda.blockDim.x) * int64(cuda.gridDim.x)
    local = int32(0)
    i = int64(S) + idx
    end = int64(E)
    while i <= end:
        local += input[i]
        i += stride
    if local != 0:
        cuda.atomic.add(output, 0, local)


def solve(inp, output, N, S, E):
    """inp is an int32 device array of length N; output is a length-1 int32
    device array."""
    if inp is None or output is None:
        return
    if S > E or N <= 0:
        zero_kernel[1, 1](output)
        cuda.synchronize()
        return
    elements = E - S + 1
    blocks = (elements + THREADS - 1) // THREADS
    if blocks > 2048:
        blocks = 2048
    zero_kernel[1, 1](output)
    sum_kernel[blocks, THREADS](inp, output, int32(S), int32(E))
    cuda.synchronize()


def _run(name, a, S, E):
    ref = int(np.sum(a[S:E + 1].astype(np.int64)))
    da = cuda.to_device(np.ascontiguousarray(a, dtype=np.int32))
    dout = cuda.device_array(1, dtype=np.int32)
    solve(da, dout, a.size, S, E)
    got = int(dout.copy_to_host()[0])
    ok = got == ref
    print(f"{name:>18} N={a.size} S={S} E={E}: got={got} ref={ref} "
          f"{'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([1, 2, 1, 3, 4], np.int32), 1, 3)
    ok &= _run("example 2", np.array([1, 2, 3, 4], np.int32), 0, 3)
    ok &= _run("single", np.array([7], np.int32), 0, 0)
    v = rng.integers(1, 11, 1000).astype(np.int32)
    ok &= _run("full 1000", v, 0, 999)
    ok &= _run("single mid", v, 500, 500)
    ok &= _run("partial", v, 10, 900)
    ok &= _run("last", v, 999, 999)
    w = rng.integers(1, 11, 1000000).astype(np.int32)
    ok &= _run("full 1e6", w, 0, 999999)
    ok &= _run("partial 1e6", w, 123456, 654321)

    # Benchmark: N = 100,000,000.
    import statistics
    import time

    N = 100000000
    a = rng.integers(1, 11, N).astype(np.int32)
    da = cuda.to_device(a)
    dout = cuda.device_array(1, dtype=np.int32)
    solve(da, dout, N, 0, N - 1)  # warm-up, also triggers the JIT compile
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(da, dout, N, 0, N - 1)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    got = int(dout.copy_to_host()[0])
    ref = int(np.sum(a.astype(np.int64)))
    print(f"Perf N={N}: {ms:.4f} ms, {N * 4 / (ms * 1e6):.1f} GB/s, "
          f"got={got} ref={ref}, {'OK' if got == ref else 'FAIL'}")
    ok &= got == ref
    print("ALL OK" if ok else "FAILURES")
