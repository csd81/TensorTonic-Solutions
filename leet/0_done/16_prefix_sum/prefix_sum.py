"""Numba CUDA port of prefix_sum.cu.

Inclusive prefix sum of a float array.

At the benchmark size (N = 250,000) the array is only 1 MB, so the cost is
dominated by how many kernels run, not by bandwidth.  The C++ version is a
single-launch decoupled-lookback scan (CUB DeviceScan style): each block
scans its own chunk, publishes its block total, and resolves the exclusive
prefix of its predecessors by walking back over published aggregates until it
finds a block whose inclusive prefix has been published.

This port keeps the same single-launch structure -- per-thread sequential
runs, a warp scan of the run totals, a warp-0 scan of the per-warp totals,
then the lookback -- plus a tiny kernel that zeroes the status array.
"""

import numpy as np
from numba import cuda, float32, int32

THREADS = 256
ITEMS = 16
BLOCK_ELEMS = THREADS * ITEMS
NWARPS = THREADS // 32
FULL = 0xFFFFFFFF

_agg = None
_pre = None
_status = None
_cap = 0


@cuda.jit(cache=True)
def zero_kernel(a, n):
    i = cuda.grid(1)
    if i < n:
        a[i] = 0


@cuda.jit(cache=True)
def scan_kernel(inp, out, agg, pre, status, n):
    warp_tot = cuda.shared.array(shape=NWARPS, dtype=float32)
    warp_pre = cuda.shared.array(shape=NWARPS, dtype=float32)
    s_pre = cuda.shared.array(shape=1, dtype=float32)
    vals = cuda.local.array(ITEMS, dtype=float32)

    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    block = cuda.blockIdx.x
    base = block * BLOCK_ELEMS + tid * ITEMS

    # Per-thread sequential inclusive scan of ITEMS consecutive elements.
    run = float32(0.0)
    for i in range(ITEMS):
        idx = base + i
        v = inp[idx] if idx < n else float32(0.0)
        run += v
        vals[i] = run

    # Warp-inclusive scan of the run totals; the exclusive value comes from
    # the previous lane's inclusive result.
    x = run
    for off in (1, 2, 4, 8, 16):
        y = cuda.shfl_up_sync(FULL, x, off)
        if lane >= off:
            x += y
    prev = cuda.shfl_up_sync(FULL, x, 1)
    if lane == 31:
        warp_tot[warp] = x
    cuda.syncthreads()

    if warp == 0:
        w = warp_tot[lane] if lane < NWARPS else float32(0.0)
        for off in (1, 2, 4, 8, 16):
            y = cuda.shfl_up_sync(FULL, w, off)
            if lane >= off:
                w += y
        if lane < NWARPS:
            warp_pre[lane] = w
    cuda.syncthreads()

    wpre = warp_pre[warp - 1] if warp > 0 else float32(0.0)
    tpre = wpre + (prev if lane > 0 else float32(0.0))
    total = warp_pre[NWARPS - 1]

    # Decoupled lookback for the block's exclusive prefix.
    if tid == 0:
        if block == 0:
            agg[0] = total
            pre[0] = total
            cuda.threadfence()
            status[0] = 2
            s_pre[0] = float32(0.0)
        else:
            agg[block] = total
            cuda.threadfence()
            status[block] = 1
            running = float32(0.0)
            i = block - 1
            while True:
                st = cuda.atomic.add(status, i, 0)
                if st == 0:
                    continue
                cuda.threadfence()
                if st == 2:
                    running = pre[i] + running
                    break
                running = agg[i] + running
                i -= 1
            pre[block] = running + total
            cuda.threadfence()
            status[block] = 2
            s_pre[0] = running
    cuda.syncthreads()

    off = s_pre[0] + tpre
    for i in range(ITEMS):
        idx = base + i
        if idx < n:
            out[idx] = off + vals[i]


def _reserve(nblocks):
    global _agg, _pre, _status, _cap
    if nblocks <= _cap:
        return
    cap = nblocks + nblocks // 4 + 1024
    _agg = cuda.device_array(cap, dtype=np.float32)
    _pre = cuda.device_array(cap, dtype=np.float32)
    _status = cuda.device_array(cap, dtype=np.int32)
    _cap = cap


def solve(input, output, N):
    if input is None or output is None:
        return
    if N <= 0:
        raise ValueError("N must be positive")

    IN = input if getattr(input, "ndim", 1) == 1 else input.reshape(N)
    OUT = output if getattr(output, "ndim", 1) == 1 else output.reshape(N)

    nblocks = (N + BLOCK_ELEMS - 1) // BLOCK_ELEMS
    _reserve(nblocks)
    zero_kernel[(nblocks + THREADS - 1) // THREADS, THREADS](_status, nblocks)
    scan_kernel[nblocks, THREADS](IN, OUT, _agg, _pre, _status, N)
    cuda.synchronize()


def _run(name, arr):
    n = arr.size
    ref = np.cumsum(arr.astype(np.float64))
    dI = cuda.to_device(np.ascontiguousarray(arr, dtype=np.float32))
    dO = cuda.device_array(n, dtype=np.float32)
    solve(dI, dO, n)
    got = dO.copy_to_host().astype(np.float64)
    err = float(np.abs(got - ref).max()) if n else 0.0
    scale = max(1.0, float(np.abs(ref).max()))
    ok = err / scale < 1e-4
    print(f"{name:>22} N={n}: max_err={err:.3e} rel={err / scale:.2e} "
          f"{'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1", np.array([1, 2, 3, 4], np.float32))
    ok &= _run("example 2", np.array([5, -2, 3, 1, -4], np.float32))
    ok &= _run("single", np.array([7.0], np.float32))
    ok &= _run("two mixed", np.array([-1.0, 2.0], np.float32))
    ok &= _run("100 ones", np.ones(100, np.float32))
    ok &= _run("1000 neg", np.full(1000, -2.0, np.float32))
    ok &= _run("2047 odd", rng.uniform(-1000, 1000, 2047).astype(np.float32))
    ok &= _run("2048 block", rng.uniform(-1000, 1000, 2048).astype(np.float32))
    ok &= _run("2049 cross-block",
               rng.uniform(-1000, 1000, 2049).astype(np.float32))
    ok &= _run("100k random",
               rng.uniform(-1000, 1000, 100000).astype(np.float32))
    ok &= _run("250k random",
               rng.uniform(-1000, 1000, 250000).astype(np.float32))

    # Performance: N = 250,000.
    import statistics
    import time

    N = 250000
    arr = rng.uniform(-1000, 1000, N).astype(np.float32)
    dI = cuda.to_device(arr)
    dO = cuda.device_array(N, dtype=np.float32)
    solve(dI, dO, N)  # warm-up
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        solve(dI, dO, N)
        times.append((time.perf_counter() - t0) * 1e6)
    us = statistics.median(times)
    ref = np.cumsum(arr.astype(np.float64))
    got = dO.copy_to_host().astype(np.float64)
    rel = float(np.abs(got - ref).max()) / max(1.0, float(np.abs(ref).max()))
    print(f"Perf N={N}: {us:.2f} us, rel={rel:.2e}, "
          f"{'OK' if rel < 1e-4 else 'FAIL'}")
    ok &= rel < 1e-4
    print("ALL OK" if ok else "FAILURES")
