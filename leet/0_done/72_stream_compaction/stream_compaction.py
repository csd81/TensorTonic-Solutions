"""Numba CUDA port of stream_compaction.cu.

Stream compaction: keep the positive elements of A in their original relative
order at the front of `out`; remaining positions stay 0 (the challenge
guarantees `out` is pre-zeroed, so only selected elements are written).

Single pass over A with a decoupled-lookback prefix sum (CUB DeviceScan):
each block owns CHUNKS chunks of 1024 elements; each thread loads four
consecutive elements, so a block-wide exclusive scan over the thread counts
reproduces the element order exactly.  The block publishes its aggregate,
looks back over preceding blocks' published aggregates until it finds an
inclusive prefix, then scatters its positives.

solve(A, N, out) keeps the challenge signature.
"""

import math

import numpy as np
from numba import cuda, float32, int32, int64

THREADS = 256
ITEMS = 4
BLOCK_ELEMS = THREADS * ITEMS
CHUNKS = 6
ELEMS_PER_BLOCK = CHUNKS * BLOCK_ELEMS
NWARPS = THREADS // 32

FLAG_AGG = np.int64(1) << 32
FLAG_PREFIX = np.int64(2) << 32
VALUE_MASK = np.int64(0xFFFFFFFF)


@cuda.jit
def compact_kernel(A, N, out, status):
    s_warp = cuda.shared.array(shape=NWARPS, dtype=int32)
    s_total = cuda.shared.array(shape=1, dtype=int32)
    s_offset = cuda.shared.array(shape=1, dtype=int32)
    s_prefix = cuda.shared.array(shape=(CHUNKS, THREADS), dtype=int32)
    s_chunk_total = cuda.shared.array(shape=CHUNKS, dtype=int32)

    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    block_base = cuda.blockIdx.x * ELEMS_PER_BLOCK

    vals = cuda.local.array(shape=(CHUNKS, ITEMS), dtype=float32)
    cnt = cuda.local.array(shape=CHUNKS, dtype=int32)

    # Load the block's elements and count the positives of each chunk.
    for c in range(CHUNKS):
        base = block_base + c * BLOCK_ELEMS + tid * ITEMS
        cc = 0
        for i in range(ITEMS):
            idx = base + i
            if idx < N:
                v = A[idx]
                vals[c, i] = v
                if v > 0.0:
                    cc += 1
            else:
                vals[c, i] = 0.0
        cnt[c] = cc

    # Per-chunk block-wide exclusive scan (rank within the chunk).
    block_total = 0
    for c in range(CHUNKS):
        val = cnt[c]
        inc = val
        for off in (1, 2, 4, 8, 16):
            n = cuda.shfl_up_sync(0xFFFFFFFF, inc, off)
            if lane >= off:
                inc += n
        if lane == 31:
            s_warp[warp] = inc
        cuda.syncthreads()
        if warp == 0:
            wv = 0
            if lane < NWARPS:
                wv = s_warp[lane]
            winc = wv
            for off in (1, 2, 4, 8, 16):
                n = cuda.shfl_up_sync(0xFFFFFFFF, winc, off)
                if lane >= off:
                    winc += n
            if lane < NWARPS:
                s_warp[lane] = winc - wv
            if lane == NWARPS - 1:
                s_total[0] = winc
        cuda.syncthreads()
        s_prefix[c, tid] = s_warp[warp] + inc - val
        if tid == 0:
            s_chunk_total[c] = s_total[0]
        block_total += s_total[0]
        cuda.syncthreads()  # s_warp / s_total are reused by the next chunk

    # Decoupled lookback: publish the aggregate, then sum the predecessors'
    # contributions until an inclusive prefix is found.
    if tid == 0:
        if cuda.blockIdx.x == 0:
            cuda.atomic.exch(status, 0, FLAG_PREFIX | np.int64(block_total))
            s_offset[0] = 0
        else:
            cuda.atomic.exch(status, cuda.blockIdx.x,
                             FLAG_AGG | np.int64(block_total))
            running = 0
            i = cuda.blockIdx.x - 1
            done = False
            while not done:
                st = cuda.atomic.add(status, i, 0)  # atomic read
                flag = st >> 32
                if flag == 0:
                    continue  # predecessor has not published yet
                running += int(st & VALUE_MASK)
                if flag == 2:
                    done = True
                else:
                    i -= 1
            cuda.atomic.exch(status, cuda.blockIdx.x,
                             FLAG_PREFIX | np.int64(running + block_total))
            s_offset[0] = running
    cuda.syncthreads()

    # Scatter the selected elements in order.
    running = s_offset[0]
    for c in range(CHUNKS):
        pos = running + s_prefix[c, tid]
        base = block_base + c * BLOCK_ELEMS + tid * ITEMS
        for i in range(ITEMS):
            if base + i < N and vals[c, i] > 0.0:
                out[pos] = vals[c, i]
                pos += 1
        running += s_chunk_total[c]


def solve(A, N, out):
    if A is None or out is None:
        return
    if N <= 0:
        raise ValueError("N must be positive")

    num_blocks = (N + ELEMS_PER_BLOCK - 1) // ELEMS_PER_BLOCK
    status = cuda.to_device(np.zeros(num_blocks, dtype=np.int64))
    compact_kernel[num_blocks, THREADS](A, N, out, status)
    cuda.synchronize()


def _reference(A):
    kept = A[A > 0.0]
    out = np.zeros(A.size, dtype=np.float32)
    out[:kept.size] = kept
    return out


def _run(name, A):
    ref = _reference(A)
    dA = cuda.to_device(A)
    # The challenge guarantees the output buffer arrives zero-initialised.
    dOut = cuda.to_device(np.zeros(A.size, dtype=np.float32))
    solve(dA, A.size, dOut)
    got = dOut.copy_to_host()
    bad = int(np.count_nonzero(got != ref))
    ok = bad == 0
    print(f"{name:>20} N={A.size} k={int((A > 0).sum())}: "
          f"{'OK' if ok else 'FAIL'} (mismatches={bad})")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    ok &= _run("example", np.array([1.0, -2.0, 3.0, 0.0, -1.0, 4.0],
                                   dtype=np.float32))
    ok &= _run("single positive", np.array([5.0], dtype=np.float32))
    ok &= _run("single negative", np.array([-5.0], dtype=np.float32))
    ok &= _run("single zero", np.array([0.0], dtype=np.float32))

    for n in (3, 1023, 1024, 1025, 4097, 100000, 1000003):
        A = rng.uniform(-1000.0, 1000.0, n).astype(np.float32)
        ok &= _run("random", A)

    A = (rng.uniform(-1000.0, 1000.0, 5000).astype(np.float32))
    A = np.abs(A) + 0.1
    ok &= _run("all positive", A.astype(np.float32))

    A = np.zeros(5000, dtype=np.float32)
    A[1::3] = -1.0
    A[2::3] = -rng.uniform(0.1, 1000.0, A[2::3].size)
    ok &= _run("none positive", A)

    A = rng.uniform(-1000.0, 1000.0, 8193).astype(np.float32)
    A[::5] = 0.0
    ok &= _run("zeros interleaved", A)

    # Performance: N = 50,000,000, median of 10 runs.
    import statistics
    import time

    N = 50000000
    A = rng.uniform(-1000.0, 1000.0, N).astype(np.float32)
    dA = cuda.to_device(A)
    dOut = cuda.to_device(np.zeros(N, dtype=np.float32))
    solve(dA, N, dOut)  # warm-up
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(dA, N, dOut)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)

    got = dOut.copy_to_host()
    ref = _reference(A)
    bad = int(np.count_nonzero(got != ref))
    k = int((A > 0).sum())
    print(f"Perf N={N} (k={k}, {100.0 * k / N:.1f}%): {ms:.3f} ms, "
          f"{'OK' if bad == 0 else 'FAIL'} (mismatches={bad})")
