"""Numba CUDA port of split_kv_attention_reduction.cu.

Flash-Decoding merge with the log-sum-exp trick:
  m_h = max_s lse[s,h], w_s = exp(lse[s,h] - m_h)
  out[h,d] = sum_s w_s * partial_out[s,h,d] / sum_s w_s
Benchmark: S=128, H=64, D=128 (4.1 MB read; the C++ kernel measures 58 us at
73 GB/s, the starter 68 us, and this port 135 us).

Same structure as the C++ kernel: one warp (32 threads) per head, the lse max
and the stable weights computed once per head with `cuda.shfl_xor_sync` and
kept in shared memory (so exp() runs 8K times instead of the starter's 1M),
and each lane accumulates the dimensions it owns in registers.

Numba has no float4 type, so a lane reads its dimensions as scalars at a
32-lane stride (each warp instruction still reads 128 contiguous bytes).  The
split loop is unrolled by two by hand, and the shared weights are exchanged
through a barrier at the one point where lanes read values written by other
lanes.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, int32
from numba.cuda import libdevice

MAX_SPLITS = 256
ALL = 0xFFFFFFFF


@cuda.jit(cache=True)
def merge_kernel(partial_out, partial_lse, output, S, H, D):
    w = cuda.shared.array(MAX_SPLITS, dtype=float32)

    h = int32(cuda.blockIdx.x)
    lane = int32(cuda.threadIdx.x)

    # Max over splits, then the stable weights and their sum -- once per head.
    m = float32(-3.4028234663852886e38)
    s = lane
    while s < S:
        v = partial_lse[s * H + h]
        w[s] = v
        if v > m:
            m = v
        s += int32(32)
    off = int32(16)
    while off > 0:
        m = libdevice.fmaxf(m, cuda.shfl_xor_sync(ALL, m, off))
        off >>= 1

    tot = float32(0.0)
    s = lane
    while s < S:
        e = libdevice.fast_expf(w[s] - m)
        w[s] = e
        tot += e
        s += int32(32)
    off = int32(16)
    while off > 0:
        tot += cuda.shfl_xor_sync(ALL, tot, off)
        off >>= 1
    inv = float32(0.0)
    if tot > 0.0:
        inv = float32(1.0) / tot

    # All lanes now read weights written by other lanes.
    cuda.syncthreads()

    v = lane
    while v < D:
        acc0 = float32(0.0)
        acc1 = float32(0.0)
        s = int32(0)
        while s + 1 < S:
            acc0 += w[s] * partial_out[(s * H + h) * D + v]
            acc1 += w[s + 1] * partial_out[((s + 1) * H + h) * D + v]
            s += int32(2)
        if s < S:
            acc0 += w[s] * partial_out[(s * H + h) * D + v]
        output[h * D + v] = (acc0 + acc1) * inv
        v += int32(32)


def solve(partial_out, partial_lse, output, S, H, D):
    """All tensors are float32 device arrays; output is (H*D)."""
    if S <= 0 or H <= 0 or D <= 0 or S > MAX_SPLITS:
        return
    merge_kernel[H, 32](partial_out, partial_lse, output, int32(S), int32(H),
                        int32(D))
    cuda.synchronize()


def _check(name, po, pl, S, H, D):
    ref = np.empty((H, D), dtype=np.float64)
    for h in range(H):
        lse = pl[:, h].astype(np.float64)
        m = lse.max()
        w = np.exp(lse - m)
        ref[h] = (w[:, None] * po[:, h, :].astype(np.float64)).sum(axis=0) / w.sum()
    dpo = cuda.to_device(np.ascontiguousarray(po, dtype=np.float32).reshape(-1))
    dpl = cuda.to_device(np.ascontiguousarray(pl, dtype=np.float32).reshape(-1))
    dout = cuda.device_array(H * D, dtype=np.float32)
    solve(dpo, dpl, dout, S, H, D)
    got = dout.copy_to_host().astype(np.float64).reshape(H, D)
    err = np.abs(got - ref)
    bad = int(np.count_nonzero(~(err <= 1e-4 + 1e-4 * np.abs(ref))))
    print(f"{name:>20} S={S} H={H} D={D}: bad={bad} "
          f"max_abs={err.max():.2e} {'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    ok &= _check("example",
                 np.array([[[1, 2], [4, 0]], [[3, 4], [0, 8]]], np.float32),
                 np.array([[0, 1.0986123], [0, 0]], np.float32), 2, 2, 2)
    for S, H, D in ((1, 1, 1), (1, 8, 16), (5, 3, 7), (17, 33, 19),
                    (64, 8, 128), (128, 64, 128), (256, 128, 256)):
        po = rng.uniform(-1, 1, (S, H, D)).astype(np.float32)
        pl = rng.uniform(-5, 20, (S, H)).astype(np.float32)
        ok &= _check(f"{S}x{H}x{D}", po, pl, S, H, D)

    # Benchmark: S=128, H=64, D=128.
    import statistics
    import time

    S, H, D = 128, 64, 128
    po = rng.uniform(-1, 1, (S, H, D)).astype(np.float32)
    pl = rng.uniform(-5, 20, (S, H)).astype(np.float32)
    ok &= _check("benchmark shape", po, pl, S, H, D)
    dpo = cuda.to_device(po.reshape(-1))
    dpl = cuda.to_device(pl.reshape(-1))
    dout = cuda.device_array(H * D, dtype=np.float32)
    solve(dpo, dpl, dout, S, H, D)  # warm-up / JIT
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        solve(dpo, dpl, dout, S, H, D)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    print(f"Perf S={S} H={H} D={D}: {ms:.4f} ms, "
          f"{S * H * D * 4 / (ms * 1e6):.1f} GB/s (partial_out only)")
    print("ALL OK" if ok else "FAILURES")
