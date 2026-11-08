"""Numba CUDA port of cross_entropy_loss.cu.

Categorical cross-entropy: loss = mean_j [ log(sum_k exp(z_jk)) - z_j,y_j ].

One warp per row, with log(sum exp) computed in a single pass using the online
softmax recurrence (so the row is read once).  Each block writes the sum of
its eight row losses to a partials array, and a tiny second kernel reduces the
partials and writes loss[0] -- no atomics and no separate zeroing launch,
which matters because kernel launches dominate at this problem size.

solve(logits, true_labels, loss, N, C) keeps the challenge signature.
"""

import math

import numpy as np
from numba import cuda, float32, int32

WARPS = 8
THREADS = WARPS * 32
NWARPS = THREADS // 32
NEG_FLT_MAX = float32(-3.4028234663852886e38)

_partials = None
_partial_capacity = 0


@cuda.jit
def ce_kernel(logits, labels, partials, N, C):
    s_warp = cuda.shared.array(shape=WARPS, dtype=float32)
    warp = cuda.threadIdx.x >> 5
    lane = cuda.threadIdx.x & 31
    row = cuda.blockIdx.x * WARPS + warp

    m = NEG_FLT_MAX
    s = float32(0.0)
    true_logit = float32(0.0)

    if row < N:
        base = row * C
        c = lane
        while c < C:
            z = logits[base + c]
            if z > m:
                s = s * math.exp(m - z) + float32(1.0)
                m = z
            else:
                s += math.exp(z - m)
            c += 32
        true_logit = logits[base + labels[row]]

    for off in (16, 8, 4, 2, 1):
        m2 = cuda.shfl_down_sync(0xFFFFFFFF, m, off)
        s2 = cuda.shfl_down_sync(0xFFFFFFFF, s, off)
        mn = m if m > m2 else m2
        s = s * math.exp(m - mn) + s2 * math.exp(m2 - mn)
        m = mn

    if lane == 0:
        if row < N:
            s_warp[warp] = (m + math.log(s)) - true_logit
        else:
            s_warp[warp] = float32(0.0)
    cuda.syncthreads()

    if cuda.threadIdx.x == 0:
        total = float32(0.0)
        for w in range(WARPS):
            total += s_warp[w]
        partials[cuda.blockIdx.x] = total


@cuda.jit
def finalize_kernel(partials, count, loss, N):
    s_warp = cuda.shared.array(shape=NWARPS, dtype=float32)
    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5

    total = float32(0.0)
    i = tid
    while i < count:
        total += partials[i]
        i += cuda.blockDim.x

    for off in (16, 8, 4, 2, 1):
        total += cuda.shfl_down_sync(0xFFFFFFFF, total, off)
    if lane == 0:
        s_warp[warp] = total
    cuda.syncthreads()
    if warp == 0:
        v = float32(0.0)
        if lane < NWARPS:
            v = s_warp[lane]
        for off in (16, 8, 4, 2, 1):
            v += cuda.shfl_down_sync(0xFFFFFFFF, v, off)
        if lane == 0:
            loss[0] = v / float32(N)


def solve(logits, true_labels, loss, N, C):
    global _partials, _partial_capacity
    if logits is None or true_labels is None or loss is None:
        return
    if N <= 0 or C <= 0:
        raise ValueError("N and C must be positive")

    # Accept flat or 2-D arrays; the kernel indexes rows linearly.
    Z = logits if getattr(logits, "ndim", 1) == 1 else logits.reshape(N * C)
    L = (true_labels if getattr(true_labels, "ndim", 1) == 1
         else true_labels.reshape(N))

    blocks = (N + WARPS - 1) // WARPS
    if blocks > _partial_capacity:
        _partials = cuda.device_array(blocks, dtype=np.float32)
        _partial_capacity = blocks

    ce_kernel[blocks, THREADS](Z, L, _partials, N, C)
    finalize_kernel[1, THREADS](_partials, blocks, loss, N)
    cuda.synchronize()


def _reference(logits, labels, N, C):
    z = logits.astype(np.float64).reshape(N, C)
    m = z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
    return float(np.mean(lse - z[np.arange(N), labels]))


def _run(name, logits, labels, N, C):
    ref = _reference(logits, labels, N, C)
    dZ = cuda.to_device(logits)
    dL = cuda.to_device(np.asarray(labels, dtype=np.int32))
    dLoss = cuda.device_array(1, dtype=np.float32)
    solve(dZ, dL, dLoss, N, C)
    got = float(dLoss.copy_to_host()[0])
    abs_err = abs(got - ref)
    rel = abs_err / max(1e-30, abs(ref))
    ok = rel < 1e-5 or abs_err < 1e-6
    print(f"{name:>20} N={N} C={C}: got={got:.8f} expected={ref:.8f} "
          f"rel={rel:.3e} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    ok &= _run("example 1",
               np.array([1.0, 2.0, 0.5, 0.1, 3.0, 1.5], dtype=np.float32),
               np.array([1, 1], dtype=np.int32), 2, 3)
    ok &= _run("example 2",
               np.array([-0.5, 1.5, 0.0, 1.0, 2.0, -1.0, 0.5, 0.5,
                         0.0, 0.0, 0.0, 0.0], dtype=np.float32),
               np.array([3, 0, 1], dtype=np.int32), 3, 4)

    for N, C in [(1, 2), (1, 1000), (5, 2), (7, 33), (100, 1000),
                 (9999, 999), (10000, 1000)]:
        logits = rng.uniform(-10.0, 10.0, (N, C)).astype(np.float32)
        labels = rng.integers(0, C, N).astype(np.int32)
        ok &= _run("random", logits, labels, N, C)

    ok &= _run("constant logits", np.full((64, 10), 1.5, dtype=np.float32),
               np.full(64, 3, dtype=np.int32), 64, 10)
    ok &= _run("extreme logits",
               np.array([-10.0, 10.0, -10.0, 10.0, -10.0, 0.0],
                        dtype=np.float32),
               np.array([0, 1], dtype=np.int32), 2, 3)

    # Performance: N = 10,000, median of 50 runs, for a few widths.
    import statistics
    import time

    N = 10000
    for C in (1000, 512, 2):
        logits = rng.uniform(-10.0, 10.0, (N, C)).astype(np.float32)
        labels = rng.integers(0, C, N).astype(np.int32)
        dZ = cuda.to_device(logits)
        dL = cuda.to_device(labels)
        dLoss = cuda.device_array(1, dtype=np.float32)
        solve(dZ, dL, dLoss, N, C)  # warm-up
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            solve(dZ, dL, dLoss, N, C)
            times.append((time.perf_counter() - t0) * 1000.0)
        ms = statistics.median(times)
        got = float(dLoss.copy_to_host()[0])
        ref = _reference(logits, labels, N, C)
        rel = abs(got - ref) / abs(ref)
        gbps = N * C * 4 / (ms * 1e6)
        print(f"Perf N={N} C={C}: {ms:.4f} ms, {gbps:.1f} GB/s, "
              f"rel={rel:.3e}, {'OK' if rel < 1e-5 else 'FAIL'}")
