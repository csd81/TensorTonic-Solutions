"""Numba CUDA port of histogramming.cu.

Histogram of 32-bit integers in [0, num_bins), num_bins <= 1024.

The problem is a single streaming read of N integers, so the C++ version is
already at the DRAM ceiling (200 MB at N = 50 M reads at ~87 GB/s and the 50 M
shared-memory atomics hide completely behind it).  This port keeps the same
shape: per-block shared histogram with atomicAdd, then one global atomic per
non-empty bin.  A 1024-entry shared array covers the maximum bin count, and a
tiny kernel zeroes the output first.

solve(input, histogram, N, num_bins) keeps the challenge signature.
"""

import numpy as np
from numba import cuda, int32

THREADS = 256
MAX_BINS = 1024


@cuda.jit
def zero_kernel(a, n):
    i = cuda.grid(1)
    if i < n:
        a[i] = 0


@cuda.jit
def histogram_kernel(inp, hist, n, nbins):
    s = cuda.shared.array(shape=MAX_BINS, dtype=int32)
    tid = cuda.threadIdx.x

    i = tid
    while i < nbins:
        s[i] = 0
        i += cuda.blockDim.x
    cuda.syncthreads()

    idx = cuda.blockIdx.x * cuda.blockDim.x + tid
    stride = cuda.blockDim.x * cuda.gridDim.x
    j = idx
    while j < n:
        cuda.atomic.add(s, inp[j], 1)
        j += stride
    cuda.syncthreads()

    i = tid
    while i < nbins:
        c = s[i]
        if c > 0:
            cuda.atomic.add(hist, i, c)
        i += cuda.blockDim.x


def solve(input, histogram, N, num_bins):
    if input is None or histogram is None:
        return
    if N <= 0 or num_bins <= 0:
        raise ValueError("N and num_bins must be positive")

    IN = input if getattr(input, "ndim", 1) == 1 else input.reshape(N)
    H = histogram if getattr(histogram, "ndim", 1) == 1 else histogram.reshape(num_bins)

    zero_kernel[(num_bins + THREADS - 1) // THREADS, THREADS](H, num_bins)
    # A grid-stride launch: enough blocks for latency hiding, not one per point.
    try:
        sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    except Exception:
        sms = 6
    blocks = sms * 32
    needed = (N + THREADS - 1) // THREADS
    if blocks > needed:
        blocks = needed
    histogram_kernel[blocks, THREADS](IN, H, N, num_bins)
    cuda.synchronize()


def _run(name, arr, nbins):
    n = arr.size
    ref = np.bincount(arr, minlength=nbins).astype(np.int32)
    dI = cuda.to_device(np.ascontiguousarray(arr, dtype=np.int32))
    dH = cuda.device_array(nbins, dtype=np.int32)
    solve(dI, dH, n, nbins)
    got = dH.copy_to_host()
    ok = np.array_equal(got, ref)
    print(f"{name:>22} N={n} bins={nbins}: {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([0, 1, 2, 1, 0], np.int32), 3)
    ok &= _run("example 2", np.array([3, 3, 3, 3], np.int32), 5)
    ok &= _run("single", np.array([0], np.int32), 1)
    ok &= _run("one bin", np.zeros(4, np.int32), 1)
    ok &= _run("zero counts", np.arange(1, 8, dtype=np.int32), 8)
    ok &= _run("1k 3 bins", rng.integers(0, 3, 1000).astype(np.int32), 3)
    ok &= _run("1k 256 bins", rng.integers(0, 256, 1000).astype(np.int32), 256)
    ok &= _run("1M 1024 bins",
               rng.integers(0, 1024, 1 << 20).astype(np.int32), 1024)
    ok &= _run("1M+1 256 bins",
               rng.integers(0, 256, 1000001).astype(np.int32), 256)
    ok &= _run("all same", np.full(10000, 7, np.int32), 256)

    # Performance: N = 50,000,000, num_bins = 256.
    import statistics
    import time

    N, BINS = 50000000, 256
    arr = rng.integers(0, BINS, N).astype(np.int32)
    dI = cuda.to_device(arr)
    dH = cuda.device_array(BINS, dtype=np.int32)
    solve(dI, dH, N, BINS)  # warm-up
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        solve(dI, dH, N, BINS)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    got = dH.copy_to_host()
    ref = np.bincount(arr, minlength=BINS).astype(np.int32)
    good = np.array_equal(got, ref)
    print(f"Perf N={N} bins={BINS}: {ms:.3f} ms, "
          f"{N * 4 / (ms * 1e6):.1f} GB/s, check={'OK' if good else 'FAIL'}")
    ok &= good
    print("ALL OK" if ok else "FAILURES")
