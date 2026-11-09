"""Numba CUDA port of row_argmax.cu.

output[b] = argmax_v logits[b,v] with ties resolved to the smallest index.
Benchmark B=256, V=128,000 (131 MB read; the C++ kernel measures 1.55 ms =
84.5 GB/s against the starter's 1.68 ms -- both are near the memory wall).

Same structure: one block (512 threads) per row, a grid-stride scan carrying
a (value, index) pair, and pair reductions through warp shuffles and a shared
array of warp results.  Scalar loads (Numba has no float4).  The merge rule
is "strictly greater, or equal with a smaller index", which gives the
first-occurrence tie break in every phase.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, int32

TPB = 512
NWARP = 16  # TPB // 32, a literal for shared-array shapes
NEG = float32(-3.4028234663852886e38)


@cuda.jit(cache=True)
def argmax_kernel(logits, out, V):
    sv = cuda.shared.array(NWARP, dtype=float32)
    si = cuda.shared.array(NWARP, dtype=int32)

    b = int32(cuda.blockIdx.x)
    tid = int32(cuda.threadIdx.x)
    base = b * V
    bv = NEG
    bi = int32(-1)
    v = tid
    while v < V:
        x = logits[base + v]
        if x > bv:
            bv = x
            bi = v
        v += int32(TPB)

    off = int32(16)
    while off > 0:
        ov = cuda.shfl_down_sync(0xFFFFFFFF, bv, off)
        oi = cuda.shfl_down_sync(0xFFFFFFFF, bi, off)
        if ov > bv or (ov == bv and oi < bi):
            bv = ov
            bi = oi
        off >>= 1
    lane = tid & 31
    warp = tid >> 5
    if lane == 0:
        sv[warp] = bv
        si[warp] = bi
    cuda.syncthreads()

    if warp == 0:
        if lane < NWARP:
            pv = sv[lane]
            pi = si[lane]
        else:
            pv = NEG
            pi = int32(-1)
        off = int32(16)
        while off > 0:
            ov = cuda.shfl_down_sync(0xFFFFFFFF, pv, off)
            oi = cuda.shfl_down_sync(0xFFFFFFFF, pi, off)
            if ov > pv or (ov == pv and oi < pi):
                pv = ov
                pi = oi
            off >>= 1
        if lane == 0:
            out[b] = pi


def solve(logits, out, B, V):
    if B <= 0 or V <= 0:
        return
    argmax_kernel[B, TPB](logits, out, int32(V))
    cuda.synchronize()


def _check(name, x, B, V):
    ref = np.empty(B, dtype=np.int64)
    xr = x.reshape(B, V)
    for b in range(B):
        ref[b] = int(np.argmax(xr[b]))  # numpy argmax returns the first max
    dl = cuda.to_device(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))
    do = cuda.device_array(B, dtype=np.int32)
    solve(dl, do, B, V)
    got = do.copy_to_host().astype(np.int64)
    bad = int(np.count_nonzero(got != ref))
    print(f"{name:>16} B={B} V={V}: bad={bad}/{B} {'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _check("example", np.array([1, 3, 2, -1, -2, -0.5, -0.5, -3], np.float32), 2, 4)
    for B, V in ((1, 1), (1, 7), (5, 33), (17, 1000), (2, 100000)):
        ok &= _check(f"{B}x{V}", rng.uniform(-100, 100, B * V).astype(np.float32), B, V)
    x = rng.uniform(-100, 100, 64 * 1000).astype(np.float32)
    for b in range(64):
        x[b * 1000 + 500] = 1000.0
        x[b * 1000 + 700] = 1000.0
        x[b * 1000 + 100] = 1000.0
    ok &= _check("ties", x, 64, 1000)
    ok &= _check("all equal", np.full(8 * 1000, 5.0, np.float32), 8, 1000)

    import statistics
    import time

    B, V = 256, 128000
    x = rng.uniform(-100, 100, B * V).astype(np.float32)
    ok &= _check("benchmark shape", x, B, V)
    dl = cuda.to_device(x)
    do = cuda.device_array(B, dtype=np.int32)
    solve(dl, do, B, V)  # warm-up / JIT
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        solve(dl, do, B, V)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    print(f"Perf B={B} V={V}: {ms:.4f} ms, {B * V * 4 / (ms * 1e6):.1f} GB/s")
    print("ALL OK" if ok else "FAILURES")
