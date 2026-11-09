"""Numba CUDA port of sorting.cu.

Sort an array of 32-bit floats in place, ascending.

The C++ version radix-sorts the IEEE-754 bit patterns: a float maps to an
order-preserving unsigned key with

    key(f) = sign(f) ? ~bits(f) : bits(f) | 0x80000000

and four 8-bit passes over the key sort everything (positive floats sort by
their raw bits, negative floats by the complement, and the inverted sign bit
puts negatives first).  The last pass converts back to floats as it stores.

This port reuses the integer radix machinery -- per-block histograms, the
two-kernel matrix scan, and the ballot-based stable ranking with a staged
write -- and does the key transform with `array.view(np.uint32)`, which Numba
supports inside kernels.

solve(data, N) keeps the challenge signature and sorts in place.
"""

import numpy as np
from numba import cuda, uint32

THREADS = 256
ITEMS = 8
BLOCK_ELEMS = THREADS * ITEMS
RADIX_BITS = 8
RADIX = 1 << RADIX_BITS
NWARPS = THREADS // 32
SCAN_ITEMS = 16
SCAN_CHUNK = THREADS * SCAN_ITEMS
FULL = 0xFFFFFFFF

_agg = None
_sums = None
_tmp0 = None
_tmp1 = None
_keys = None
_cover = 0
_bcover = 0


@cuda.jit(device=True)
def match_digit(d):
    """Mask of lanes sharing this lane's digit."""
    m = uint32(FULL)
    for b in range(RADIX_BITS):
        bit = cuda.ballot_sync(FULL, (d >> b) & 1)
        if (d >> b) & 1:
            m &= bit
        else:
            m &= ~bit
    return m


# ------------------------------------------------------------------ stage 1

@cuda.jit
def transform_kernel(src, dst, n):
    """float array -> order-preserving uint32 keys."""
    i = cuda.grid(1)
    if i < n:
        b = src.view(np.uint32)[i]
        if b & 0x80000000:
            dst[i] = (~b) & 0xFFFFFFFF
        else:
            dst[i] = b | 0x80000000


@cuda.jit
def untransform_kernel(keys, out, n):
    """uint32 keys -> floats, written in place over the original array."""
    i = cuda.grid(1)
    if i < n:
        k = keys[i]
        if k & 0x80000000:
            out.view(np.uint32)[i] = k & 0x7FFFFFFF
        else:
            out.view(np.uint32)[i] = (~k) & 0xFFFFFFFF


@cuda.jit
def hist_kernel(src, agg, N, shift, nblocks):
    hist = cuda.shared.array(shape=RADIX, dtype=uint32)
    tid = cuda.threadIdx.x
    if tid < RADIX:
        hist[tid] = 0
    cuda.syncthreads()

    base = cuda.blockIdx.x * BLOCK_ELEMS
    for j in range(ITEMS):
        idx = base + j * THREADS + tid
        if idx < N:
            cuda.atomic.add(hist, (src[idx] >> shift) & (RADIX - 1), 1)
    cuda.syncthreads()

    if tid < RADIX:
        agg[tid * nblocks + cuda.blockIdx.x] = hist[tid]


# ------------------------------------------------------------------ stage 2

@cuda.jit
def scan_blocks_kernel(data, sums, n):
    s_warp = cuda.shared.array(shape=NWARPS, dtype=uint32)
    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    base = cuda.blockIdx.x * SCAN_CHUNK + tid * SCAN_ITEMS

    run = cuda.local.array(SCAN_ITEMS, dtype=uint32)
    s = uint32(0)
    for j in range(SCAN_ITEMS):
        idx = base + j
        v = data[idx] if idx < n else uint32(0)
        run[j] = s
        s += v

    x = s
    for off in (1, 2, 4, 8, 16):
        y = cuda.shfl_up_sync(FULL, x, off)
        if lane >= off:
            x += y
    if lane == 31:
        s_warp[warp] = x
    cuda.syncthreads()
    if warp == 0:
        w = s_warp[lane] if lane < NWARPS else uint32(0)
        for off in (1, 2, 4, 8, 16):
            y = cuda.shfl_up_sync(FULL, w, off)
            if lane >= off:
                w += y
        if lane < NWARPS:
            s_warp[lane] = w
    cuda.syncthreads()

    wbase = s_warp[warp - 1] if warp > 0 else uint32(0)
    tbase = wbase + (x - s)
    for j in range(SCAN_ITEMS):
        idx = base + j
        if idx < n:
            data[idx] = run[j] + tbase
    if tid == THREADS - 1:
        sums[cuda.blockIdx.x] = wbase + x


@cuda.jit
def scan_sums_kernel(sums, m):
    s_warp = cuda.shared.array(shape=NWARPS, dtype=uint32)
    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    runs = (m + THREADS - 1) // THREADS
    base = tid * runs

    s = uint32(0)
    for j in range(runs):
        if base + j < m:
            s += sums[base + j]

    x = s
    for off in (1, 2, 4, 8, 16):
        y = cuda.shfl_up_sync(FULL, x, off)
        if lane >= off:
            x += y
    if lane == 31:
        s_warp[warp] = x
    cuda.syncthreads()
    if warp == 0:
        w = s_warp[lane] if lane < NWARPS else uint32(0)
        for off in (1, 2, 4, 8, 16):
            y = cuda.shfl_up_sync(FULL, w, off)
            if lane >= off:
                w += y
        if lane < NWARPS:
            s_warp[lane] = w
    cuda.syncthreads()

    wbase = s_warp[warp - 1] if warp > 0 else uint32(0)
    acc = wbase + (x - s)
    for j in range(runs):
        if base + j < m:
            v = sums[base + j]
            sums[base + j] = acc
            acc += v


def scan_device(data, sums, n):
    nb = (n + SCAN_CHUNK - 1) // SCAN_CHUNK
    scan_blocks_kernel[nb, THREADS](data, sums, n)
    scan_sums_kernel[1, THREADS](sums, nb)


# ------------------------------------------------------------------ stage 3

@cuda.jit
def scatter_kernel(src, dst, off, sums, N, shift, nblocks):
    warp_hist = cuda.shared.array(shape=(NWARPS, RADIX), dtype=uint32)
    block_hist = cuda.shared.array(shape=RADIX, dtype=uint32)
    block_pref = cuda.shared.array(shape=RADIX, dtype=uint32)
    block_off = cuda.shared.array(shape=RADIX, dtype=uint32)
    stage = cuda.shared.array(shape=BLOCK_ELEMS, dtype=uint32)

    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    block = cuda.blockIdx.x

    for i in range(tid, NWARPS * RADIX, THREADS):
        warp_hist[i // RADIX, i % RADIX] = 0
    if tid < RADIX:
        idx = tid * nblocks + block
        block_off[tid] = off[idx] + sums[idx // SCAN_CHUNK]
    cuda.syncthreads()

    # (Numba's two-argument min() does not type in CUDA kernels.)
    nvalid = BLOCK_ELEMS
    if N - block * BLOCK_ELEMS < nvalid:
        nvalid = N - block * BLOCK_ELEMS

    vals = cuda.local.array(ITEMS, dtype=uint32)
    dig = cuda.local.array(ITEMS, dtype=uint32)
    rank = cuda.local.array(ITEMS, dtype=uint32)
    valid = cuda.local.array(ITEMS, dtype=np.int32)
    for i in range(ITEMS):
        pos = warp * (ITEMS * 32) + i * 32 + lane
        if pos < nvalid:
            valid[i] = 1
            vals[i] = src[block * BLOCK_ELEMS + pos]
            dig[i] = (vals[i] >> shift) & (RADIX - 1)
        else:
            valid[i] = 0
            vals[i] = 0
            dig[i] = 0

    for i in range(ITEMS):
        vmask = cuda.ballot_sync(FULL, valid[i] == 1)
        md = match_digit(dig[i])
        m = (md & vmask) if valid[i] == 1 else uint32(0)
        prev = warp_hist[warp, dig[i]]
        if valid[i] == 1:
            rank[i] = prev + cuda.popc(m & cuda.lanemask_lt())
            if lane == cuda.ffs(m) - 1:
                warp_hist[warp, dig[i]] = prev + cuda.popc(m)
        cuda.syncwarp()
    cuda.syncthreads()

    if tid < RADIX:
        s = uint32(0)
        for w in range(NWARPS):
            c = warp_hist[w, tid]
            warp_hist[w, tid] = s
            s += c
        block_hist[tid] = s
    cuda.syncthreads()

    # Exclusive scan of the block histogram over the digits.
    if tid < RADIX:
        block_pref[tid] = block_hist[tid]
    cuda.syncthreads()
    d = 1
    while d < RADIX:
        v = uint32(0)
        if tid < RADIX and tid >= d:
            v = block_pref[tid - d]
        cuda.syncthreads()
        if tid < RADIX:
            block_pref[tid] += v
        cuda.syncthreads()
        d <<= 1
    if tid < RADIX:
        block_pref[tid] -= block_hist[tid]
    cuda.syncthreads()

    for i in range(ITEMS):
        if valid[i] == 1:
            stage[block_pref[dig[i]] + warp_hist[warp, dig[i]] + rank[i]] = vals[i]
    cuda.syncthreads()

    for j in range(ITEMS):
        p = tid + j * THREADS
        if p < nvalid:
            v = stage[p]
            d = (v >> shift) & (RADIX - 1)
            dst[block_off[d] + (p - block_pref[d])] = v


def _reserve(n, nblocks):
    global _agg, _sums, _tmp0, _tmp1, _keys, _cover, _bcover
    if n <= _cover and nblocks <= _bcover:
        return
    cap = n + n // 4 + 1024
    bcap = nblocks + nblocks // 4 + 64
    _agg = cuda.device_array(bcap * RADIX, dtype=np.uint32)
    _sums = cuda.device_array(bcap, dtype=np.uint32)
    _tmp0 = cuda.device_array(cap, dtype=np.uint32)
    _tmp1 = cuda.device_array(cap, dtype=np.uint32)
    _keys = cuda.device_array(cap, dtype=np.uint32)
    _cover = cap
    _bcover = bcap


def solve(data, N):
    if data is None:
        return
    if N <= 0:
        raise ValueError("N must be positive")

    A = data if getattr(data, "ndim", 1) == 1 else data.reshape(N)

    nblocks = (N + BLOCK_ELEMS - 1) // BLOCK_ELEMS
    _reserve(N, nblocks)
    matrix = nblocks * RADIX
    blocks = (N + THREADS - 1) // THREADS

    transform_kernel[blocks, THREADS](A, _keys, N)

    PASSS = (32 + RADIX_BITS - 1) // RADIX_BITS
    src = _keys
    for p in range(PASSS):
        shift = p * RADIX_BITS
        dst = _tmp0 if p % 2 == 0 else _tmp1
        hist_kernel[nblocks, THREADS](src, _agg, N, shift, nblocks)
        scan_device(_agg, _sums, matrix)
        scatter_kernel[nblocks, THREADS](src, dst, _agg, _sums, N, shift,
                                         nblocks)
        src = dst
    untransform_kernel[blocks, THREADS](src, A, N)
    cuda.synchronize()


def _reference(a):
    return np.sort(a.astype(np.float32))


def _run(name, arr):
    n = arr.size
    ref = _reference(arr)
    d = cuda.to_device(np.ascontiguousarray(arr, dtype=np.float32))
    solve(d, n)
    got = d.copy_to_host()
    ok = np.array_equal(got, ref)
    print(f"{name:>22} N={n}: {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example", np.array([5, 2, 8, 1, 9, 4], np.float32))
    ok &= _run("single", np.array([7.0], np.float32))
    ok &= _run("two", np.array([2.0, 1.0], np.float32))
    ok &= _run("sorted", np.array([1, 2, 3, 4], np.float32))
    ok &= _run("reversed", np.array([4, 3, 2, 1], np.float32))
    ok &= _run("duplicates", np.array([3, 3, 3, 1, 1], np.float32))
    ok &= _run("zeros", np.array([0.0, -0.0, 1.0, -1.0], np.float32))
    ok &= _run("negatives",
               np.array([-5.5, -100.0, 3.25, -0.5, 0.0, 1e-8, 1e8], np.float32))
    ok &= _run("1k random", rng.uniform(-1000, 1000, 1000).astype(np.float32))
    ok &= _run("1k positives", rng.uniform(0, 1, 1000).astype(np.float32))
    ok &= _run("1023 odd", rng.uniform(-100, 100, 1023).astype(np.float32))
    ok &= _run("1024 pow2", rng.uniform(-100, 100, 1024).astype(np.float32))
    ok &= _run("100k random", rng.uniform(-1e6, 1e6, 100000).astype(np.float32))

    # Performance: N = 1,000,000.
    import statistics
    import time

    N = 1000000
    arr = rng.uniform(-1000, 1000, N).astype(np.float32)
    d = cuda.to_device(arr)
    solve(d, N)  # warm-up
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(d, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    got = d.copy_to_host()
    ok &= bool(np.array_equal(got, np.sort(arr)))
    print(f"Perf N={N}: {ms:.3f} ms, check={'OK' if np.array_equal(got, np.sort(arr)) else 'FAIL'}")
    print("ALL OK" if ok else "FAILURES")
