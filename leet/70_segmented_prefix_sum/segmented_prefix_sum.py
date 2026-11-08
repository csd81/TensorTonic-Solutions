"""Numba CUDA port of segmented_prefix_sum.cu.

Exclusive segmented prefix sum: output[i] is the sum of the values before i
inside i's segment (0 at a flag), where flags[i] == 1 starts a new segment.

Same algorithm as the CUDA C++ kernel -- one pass over the data with a
decoupled-lookback scan (CUB DeviceScan style) generalised to the segmented
operator

    seg_op(a, b) = ( b.flag ? b.val : a.val + b.val , a.flag | b.flag )

but laid out for Numba's code generation.  Numba cannot keep a per-thread
array of 16 values in registers, and having each thread own 16 *consecutive*
elements makes every warp load a stride-16 gather (measured 16.3 ms for the
reads alone, versus 5.9 ms coalesced).  So the block is split into CHUNKS
chunks of THREADS elements: chunk c is [base + c*THREADS, +THREADS), every
load and store is fully coalesced, and each chunk gets its own warp scan.
All chunks share the same two barriers, so the block still costs only three
__syncthreads(): one after publishing the per-warp aggregates, one after warp
0 has scanned them, and one after the lookback.

The value of each element is needed twice -- once for the aggregate that feeds
the scan and once to emit the output.  Rather than hold it (impossible in
Numba), the second phase re-reads it; the block's data is only 32 KB and is
still in L1/L2, so DRAM traffic stays at read-once + write-once.

Accumulation is float64; the output is float32 as specified.

solve(values, flags, output, N) keeps the challenge signature.  All three
arrays may be Numba device arrays (preferred) or host NumPy arrays.
"""

import numpy as np
from numba import cuda, float32, float64, int32

THREADS = 256
CHUNKS = 8
IPT = 4  # consecutive elements per thread per chunk
BLOCK_ELEMS = THREADS * IPT * CHUNKS
NWARPS = THREADS // 32
SV_SHAPE = CHUNKS * NWARPS  # per-chunk per-warp aggregates

# Cached lookback scratch (aggregate / inclusive prefix / status per block).
_agg_val = None
_agg_flag = None
_pre_val = None
_pre_flag = None
_status = None
_zeros = None
_capacity = 0


@cuda.jit
def zero_kernel(a, n):
    i = cuda.grid(1)
    if i < n:
        a[i] = 0


@cuda.jit
def segscan_kernel(values, flags, output, agg_val, agg_flag, pre_val, pre_flag,
                   status, N):
    s_val = cuda.shared.array(shape=SV_SHAPE, dtype=float64)
    s_flag = cuda.shared.array(shape=SV_SHAPE, dtype=int32)
    s_blk = cuda.shared.array(shape=2, dtype=float64)  # [0]=val, [1]=flag

    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    block = cuda.blockIdx.x
    block_base = block * BLOCK_ELEMS

    # ---- Phase 1: warp scan of each chunk; publish per-warp aggregates ----
    for c in range(CHUNKS):
        cbase = block_base + c * (THREADS * IPT) + tid * IPT
        xv = float64(0.0)
        xf = int32(0)
        for i in range(IPT):
            idx = cbase + i
            if idx < N:
                v = float64(values[idx])
                if flags[idx] != 0:
                    xv = v
                    xf = int32(1)
                else:
                    xv += v
        for off in (1, 2, 4, 8, 16):
            ov = cuda.shfl_up_sync(0xFFFFFFFF, xv, off)
            of = cuda.shfl_up_sync(0xFFFFFFFF, xf, off)
            if lane >= off:
                if xf == 0:  # seg_op((ov,of),(xv,xf)) with no reset here
                    xv = ov + xv
                xf = of | xf
        if lane == 31:
            s_val[c * NWARPS + warp] = xv
            s_flag[c * NWARPS + warp] = xf
    cuda.syncthreads()

    # ---- Warp 0 scans the per-warp aggregates of every chunk --------------
    if warp == 0:
        for c in range(CHUNKS):
            wv = float64(0.0)
            wf = int32(0)
            if lane < NWARPS:
                wv = s_val[c * NWARPS + lane]
                wf = s_flag[c * NWARPS + lane]
            for off in (1, 2, 4, 8, 16):
                ov = cuda.shfl_up_sync(0xFFFFFFFF, wv, off)
                of = cuda.shfl_up_sync(0xFFFFFFFF, wf, off)
                if lane >= off:
                    if wf == 0:
                        wv = ov + wv
                    wf = of | wf
            if lane < NWARPS:
                s_val[c * NWARPS + lane] = wv
                s_flag[c * NWARPS + lane] = wf
    cuda.syncthreads()

    # s_val[c*NWARPS + NWARPS-1] is chunk c's aggregate.
    bav = s_val[NWARPS - 1]
    baf = s_flag[NWARPS - 1]
    for c in range(1, CHUNKS):
        av = s_val[c * NWARPS + NWARPS - 1]
        af = s_flag[c * NWARPS + NWARPS - 1]
        if af == 0:
            bav = bav + av
        else:
            bav = av
        baf = baf | af

    # ---- Decoupled lookback (thread 0) ------------------------------------
    if tid == 0:
        if block == 0:
            agg_val[0] = bav
            agg_flag[0] = baf
            # status 2 entries hold the INCLUSIVE prefix (prefix + aggregate)
            pre_val[0] = bav
            pre_flag[0] = baf
            cuda.threadfence()
            status[0] = 2
            s_blk[0] = float64(0.0)
            s_blk[1] = float64(0.0)
        else:
            agg_val[block] = bav
            agg_flag[block] = baf
            cuda.threadfence()
            status[block] = 1

            run_v = float64(0.0)
            run_f = int32(0)
            i = block - 1
            while True:
                st = cuda.atomic.add(status, i, 0)  # coherent read
                if st == 0:
                    continue
                cuda.threadfence()
                iv = agg_val[i]
                iff = agg_flag[i]
                if st == 2:
                    iv = pre_val[i]
                    iff = pre_flag[i]
                # run = seg_op(item, run)
                if run_f == 0:
                    run_v = iv + run_v
                run_f = iff | run_f
                if st == 2:
                    break
                i -= 1
            # Publish this block's inclusive prefix.
            if baf == 0:
                pre_val[block] = run_v + bav
            else:
                pre_val[block] = bav
            pre_flag[block] = run_f | baf
            cuda.threadfence()
            status[block] = 2
            s_blk[0] = run_v
            s_blk[1] = float64(run_f)
    cuda.syncthreads()

    # ---- Phase 2: chunk prefixes and the output ---------------------------
    cpv = s_blk[0]
    cpf = int32(s_blk[1])
    for c in range(CHUNKS):
        cbase = block_base + c * (THREADS * IPT) + tid * IPT
        xv = float64(0.0)
        xf = int32(0)
        for i in range(IPT):
            idx = cbase + i
            if idx < N:
                v = float64(values[idx])  # re-read: L1/L2-hot
                if flags[idx] != 0:
                    xv = v
                    xf = int32(1)
                else:
                    xv += v
        for off in (1, 2, 4, 8, 16):
            ov = cuda.shfl_up_sync(0xFFFFFFFF, xv, off)
            of = cuda.shfl_up_sync(0xFFFFFFFF, xf, off)
            if lane >= off:
                if xf == 0:
                    xv = ov + xv
                xf = of | xf
        # Thread exclusive within the chunk.
        wpv = float64(0.0)
        wpf = int32(0)
        if warp > 0:
            wpv = s_val[c * NWARPS + warp - 1]
            wpf = s_flag[c * NWARPS + warp - 1]
        pv = cuda.shfl_up_sync(0xFFFFFFFF, xv, 1)
        pf = cuda.shfl_up_sync(0xFFFFFFFF, xf, 1)
        exv = wpv
        exf = wpf
        if lane > 0:
            if pf == 0:
                exv = wpv + pv
            else:
                exv = pv
            exf = wpf | pf
        # run = seg_op(chunk_prefix, exclusive)
        if exf == 0:
            run = cpv + exv
        else:
            run = exv
        for i in range(IPT):
            idx = cbase + i
            if idx < N:
                if flags[idx] != 0:
                    output[idx] = float32(0.0)
                    run = float64(values[idx])
                else:
                    output[idx] = float32(run)
                    run += float64(values[idx])
        # chunk_prefix = seg_op(chunk_prefix, chunk_aggregate)
        cav = s_val[c * NWARPS + NWARPS - 1]
        caf = s_flag[c * NWARPS + NWARPS - 1]
        if caf == 0:
            cpv = cpv + cav
        else:
            cpv = cav
        cpf = cpf | caf


def _reserve(num_blocks):
    global _agg_val, _agg_flag, _pre_val, _pre_flag, _status, _zeros, _capacity
    if num_blocks <= _capacity:
        return
    cap = num_blocks + num_blocks // 4 + 1024
    _agg_val = cuda.device_array(cap, dtype=np.float64)
    _agg_flag = cuda.device_array(cap, dtype=np.int32)
    _pre_val = cuda.device_array(cap, dtype=np.float64)
    _pre_flag = cuda.device_array(cap, dtype=np.int32)
    _status = cuda.device_array(cap, dtype=np.int32)
    _zeros = cuda.to_device(np.zeros(cap, dtype=np.int32))
    _capacity = cap


def solve(values, flags, output, N):
    if values is None or flags is None or output is None:
        return
    if N <= 0:
        raise ValueError("N must be positive")

    V = values if getattr(values, "ndim", 1) == 1 else values.reshape(N)
    F = flags if getattr(flags, "ndim", 1) == 1 else flags.reshape(N)
    O = output if getattr(output, "ndim", 1) == 1 else output.reshape(N)

    num_blocks = (N + BLOCK_ELEMS - 1) // BLOCK_ELEMS
    _reserve(num_blocks)
    zero_kernel[(num_blocks + 255) // 256, 256](_status, num_blocks)
    segscan_kernel[num_blocks, THREADS](V, F, O, _agg_val, _agg_flag, _pre_val,
                                        _pre_flag, _status, N)
    cuda.synchronize()


def _reference(values, flags):
    n = values.size
    ref = np.zeros(n, dtype=np.float64)
    acc = 0.0
    for i in range(n):
        if flags[i]:
            ref[i] = 0.0
            acc = float(values[i])
        else:
            ref[i] = acc
            acc += float(values[i])
    return ref


def _run(name, values, flags):
    n = values.size
    ref = _reference(values, flags)
    dV = cuda.to_device(np.ascontiguousarray(values, dtype=np.float32))
    dF = cuda.to_device(np.ascontiguousarray(flags, dtype=np.int32))
    dO = cuda.device_array(n, dtype=np.float32)
    solve(dV, dF, dO, n)
    got = dO.copy_to_host().astype(np.float64)
    err = np.abs(got - ref)
    max_err = float(err.max()) if n else 0.0
    max_ref = float(np.abs(ref).max()) if n else 0.0
    rel = max_err / max(1e-30, max_ref)
    ok = rel < 1e-5
    print(f"{name:>20} N={n}: max_err={max_err:.3e} rel={rel:.3e} "
          f"{'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    ok &= _run("example", np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
               np.array([1, 0, 0, 1, 0, 1], dtype=np.int32))
    ok &= _run("single element", np.array([5.0], dtype=np.float32),
               np.array([1], dtype=np.int32))

    def random_case(n, p, name):
        v = rng.uniform(-100.0, 100.0, n).astype(np.float32)
        f = (rng.random(n) < p).astype(np.int32)
        f[0] = 1
        return _run(name, v, f)

    ok &= random_case(1000, 0.05, "1k few segments")
    ok &= random_case(1000, 0.5, "1k many segments")
    ok &= random_case(4097, 0.001, "4k long segments")
    ok &= random_case(100000, 0.01, "100k")
    ok &= random_case(1 << 20, 0.002, "1M")
    ok &= _run("all segments",
               rng.uniform(-100.0, 100.0, 10000).astype(np.float32),
               np.ones(10000, dtype=np.int32))
    v = rng.uniform(-100.0, 100.0, 1000000).astype(np.float32)
    f = np.zeros(1000000, dtype=np.int32)
    f[0] = 1
    ok &= _run("single segment", v, f)

    # Performance: N = 50,000,000, two segment densities.
    import statistics
    import time

    N = 50000000
    for p in (0.001, 0.1):
        values = rng.uniform(-100.0, 100.0, N).astype(np.float32)
        flags = (rng.random(N) < p).astype(np.int32)
        flags[0] = 1
        dV = cuda.to_device(values)
        dF = cuda.to_device(flags)
        dO = cuda.device_array(N, dtype=np.float32)
        solve(dV, dF, dO, N)  # warm-up
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            solve(dV, dF, dO, N)
            times.append((time.perf_counter() - t0) * 1000.0)
        ms = statistics.median(times)
        out = dO.copy_to_host()
        max_err = 0.0
        max_ref = 0.0
        for i in range(0, N, 1000003):
            j = i
            acc = 0.0
            while j > 0 and flags[j] == 0:
                j -= 1
            for k in range(j, i):
                acc += float(values[k])
            max_err = max(max_err, abs(float(out[i]) - acc))
            max_ref = max(max_ref, abs(acc))
        rel = max_err / max(1e-30, max_ref)
        gbps = 3.0 * N * 4 / (ms * 1e6)
        print(f"Perf N={N} p={p}: {ms:.3f} ms, {gbps:.1f} GB/s, "
              f"rel={rel:.2e} {'OK' if rel < 1e-5 else 'FAIL'}")
        ok &= rel < 1e-5
    print("ALL OK" if ok else "FAILURES")
