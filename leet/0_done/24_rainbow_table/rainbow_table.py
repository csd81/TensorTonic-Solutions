"""Numba CUDA port of rainbow_table.cu.

Apply R rounds of FNV-1a to each 32-bit element: out[i] = fnv1a^R(in[i]).

At N = 5,000,000 the traffic is 40 MB, so R = 1 is memory-bound (~0.49 ms,
the copy floor).  As R grows the dependent multiply chain dominates: the C++
version interleaves four independent chains per thread (one int4 load) and
measures 0.90 ms at R = 10 and 9.27 ms at R = 100, versus 1.07/9.55 ms for
the scalar one-element-per-thread form.

Numba has no int4 type, so this port keeps one element per thread; the R
rounds are a plain loop.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, uint32

THREADS = 256
P = np.uint32(16777619)
OFFSET = np.uint32(2166136261)


@cuda.jit(cache=True)
def rainbow_kernel(input, output, n, R):
    i = cuda.grid(1)
    if i < n:
        v = uint32(input[i])
        for _ in range(R):
            h = uint32(2166136261)
            h = (h ^ (v & uint32(0xFF))) * uint32(16777619)
            h = (h ^ ((v >> uint32(8)) & uint32(0xFF))) * uint32(16777619)
            h = (h ^ ((v >> uint32(16)) & uint32(0xFF))) * uint32(16777619)
            # Mask the top byte too: Numba carries these products in 64-bit
            # registers, so an unmasked (v >> 24) would pull in bits 32+ of a
            # previous round's product and corrupt the next round.
            h = (h ^ ((v >> uint32(24)) & uint32(0xFF))) * uint32(16777619)
            v = h
        output[i] = v


def solve(input, output, N, R):
    """input is an int32 device array, output a uint32 device array."""
    if N <= 0:
        return
    if input is None or output is None:
        return
    blocks = (N + THREADS - 1) // THREADS
    rainbow_kernel[blocks, THREADS](input, output, N, R)
    cuda.synchronize()


def _ref(a, R):
    v = a.astype(np.uint32).copy()
    for _ in range(R):
        h = np.full_like(v, OFFSET)
        for b in range(4):
            h = (h ^ ((v >> np.uint32(8 * b)) & np.uint32(0xFF))) * P
        v = h
    return v


def _run(name, a, R):
    n = a.size
    ref = _ref(a, R)
    da = cuda.to_device(np.ascontiguousarray(a, dtype=np.int32))
    dout = cuda.device_array(n, dtype=np.uint32)
    solve(da, dout, n, R)
    got = dout.copy_to_host()
    bad = int(np.count_nonzero(got != ref))
    ok = bad == 0
    print(f"{name:>24} N={n} R={R}: mismatches={bad} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([123, 456, 789], np.int32), 2)
    ok &= _run("example 2", np.array([0, 1, 2147483647], np.int32), 3)
    ok &= _run("single zero, R=1", np.array([0], np.int32), 1)
    ok &= _run("single max, R=100", np.array([2147483647], np.int32), 100)
    ok &= _run("all zero, R=100", np.zeros(64, np.int32), 100)
    for n in (2, 3, 5, 1023, 1025, 4099, 100000):
        ok &= _run(f"random {n}", rng.integers(0, 2147483648, n, dtype=np.int64)
                   .astype(np.int32), 7)

    # Benchmark: N = 5,000,000.
    import statistics
    import time

    N = 5000000
    a = rng.integers(0, 2147483648, N, dtype=np.int64).astype(np.int32)
    da = cuda.to_device(a)
    dout = cuda.device_array(N, dtype=np.uint32)
    bytes_moved = 2 * N * 4
    for R in (1, 10, 100):
        solve(da, dout, N, R)  # warm-up
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            solve(da, dout, N, R)
            times.append((time.perf_counter() - t0) * 1000.0)
        ms = statistics.median(times)
        bad = int(np.count_nonzero(dout.copy_to_host() != _ref(a, R)))
        print(f"Perf N={N} R={R}: {ms:.4f} ms, "
              f"{bytes_moved / (ms * 1e6):.1f} GB/s, mismatches={bad}, "
              f"{'OK' if bad == 0 else 'FAIL'}")
        ok &= bad == 0
    print("ALL OK" if ok else "FAILURES")
