"""Numba CUDA port of moe_token_dispatch.cu.

Given x [T, D] and expert_idx [T], fill dispatched_x[e, :token_counts[e], :]
with the tokens routed to expert e in ascending token order, and fill
token_counts.  Benchmark: T=16384, D=512, E=8 (64 MB of copy traffic; the
copy floor is ~0.8 ms).

The C++ kernel assigns stable slots with one warp per expert: each window of
32 consecutive tokens is ballot-compacted (`__ballot_sync(exp == e)`, lane
rank = popcount of the mask below the lane, running count += popcount), then
a second kernel copies one token row per warp with float4 loads and stores.

Numba exposes `cuda.ballot_sync` and `cuda.popc`, so the slot kernel is the
same algorithm here.  Numba has no float4 data type, so the copy kernel moves
scalars (each warp still reads 128 contiguous bytes per instruction, so the
accesses stay coalesced); this is the main reason the port is slower.
Indices into dispatched_x are built in int64 because the largest offset
(E*T*D) can exceed 2^31.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, int32, int64

WARP = 32
THREADS = 256
ALL = 0xFFFFFFFF

_slots = None


@cuda.jit(cache=True)
def slot_kernel(expert_idx, token_slots, token_counts, T):
    e = int32(cuda.blockIdx.x)
    lane = int32(cuda.threadIdx.x)
    running = int32(0)
    windows = (T + 31) // 32
    w = int32(0)
    while w < windows:
        i = w * int32(32) + lane
        exp = int32(-1)
        if i < T:
            exp = expert_idx[i]
        mask = cuda.ballot_sync(ALL, exp == e)
        rank = cuda.popc(mask & ((int32(1) << lane) - int32(1)))
        if exp == e:
            token_slots[i] = running + rank
        running += cuda.popc(mask)
        w += int32(1)
    if lane == 0:
        token_counts[e] = running


@cuda.jit(cache=True)
def dispatch_kernel(x, expert_idx, token_slots, dispatched_x, T, D, capacity):
    warp = int32(cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) >> 5
    lane = int32(cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) & 31
    if warp < T:
        exp = expert_idx[warp]
        slot = token_slots[warp]
        src_base = int64(warp) * int64(D)
        dst_base = (int64(exp) * int64(capacity) + int64(slot)) * int64(D)
        i = lane
        while i < D:
            dispatched_x[dst_base + i] = x[src_base + i]
            i += int32(32)


def solve(x, expert_idx, dispatched_x, token_counts, T, D, E, capacity):
    """All arguments are device arrays; token_counts is int32[E]."""
    global _slots
    if T <= 0 or D <= 0 or E <= 0:
        return
    if _slots is None or _slots.size < T:
        _slots = cuda.device_array(T, dtype=np.int32)
    slot_kernel[E, WARP](expert_idx, _slots, token_counts, int32(T))
    blocks = (T + THREADS // WARP - 1) // (THREADS // WARP)
    dispatch_kernel[blocks, THREADS](x, expert_idx, _slots, dispatched_x,
                                     int32(T), int32(D), int32(capacity))
    cuda.synchronize()


def _check(name, x, idx, T, D, E, capacity):
    counts_ref = np.zeros(E, dtype=np.int64)
    groups = [[] for _ in range(E)]
    for t in range(T):
        groups[idx[t]].append(t)
        counts_ref[idx[t]] += 1

    dx = cuda.to_device(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))
    didx = cuda.to_device(np.ascontiguousarray(idx, dtype=np.int32))
    dout = cuda.device_array(E * capacity * D, dtype=np.float32)
    dcounts = cuda.device_array(E, dtype=np.int32)
    solve(dx, didx, dout, dcounts, T, D, E, capacity)
    counts = dcounts.copy_to_host()
    out = dout.copy_to_host().reshape(E, capacity, D)

    bad = 0
    for e in range(E):
        if counts[e] != counts_ref[e]:
            bad += 1
            continue
        for s, t in enumerate(groups[e]):
            if not np.array_equal(out[e, s], x[t]):
                bad += 1
                break
    print(f"{name:>22} T={T} D={D} E={E}: bad={bad} "
          f"{'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    ok &= _check("example",
                 np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
                          np.float32),
                 np.array([0, 1, 0, 1], np.int32), 4, 3, 2, 4)
    for T, D, E in ((1, 1, 2), (1000, 7, 8), (1000, 512, 8), (5000, 64, 3),
                    (65536, 4, 64)):
        x = rng.uniform(-1, 1, (T, D)).astype(np.float32)
        idx = rng.integers(0, E, T).astype(np.int32)
        ok &= _check(f"{T}x{D} e{E}", x, idx, T, D, E, T)
    x = rng.uniform(-1, 1, (500, 33)).astype(np.float32)
    ok &= _check("all expert 0", x, np.zeros(500, np.int32), 500, 33, 4, 500)
    ok &= _check("round robin", x, (np.arange(500) % 4).astype(np.int32), 500,
                 33, 4, 500)

    # Benchmark: T = 16,384, D = 512, E = 8.
    import statistics
    import time

    T, D, E = 16384, 512, 8
    x = rng.uniform(-1, 1, (T, D)).astype(np.float32)
    idx = rng.integers(0, E, T).astype(np.int32)
    ok &= _check("benchmark shape", x, idx, T, D, E, T)

    dx = cuda.to_device(x.reshape(-1))
    didx = cuda.to_device(idx)
    dout = cuda.device_array(E * T * D, dtype=np.float32)
    dcounts = cuda.device_array(E, dtype=np.int32)
    solve(dx, didx, dout, dcounts, T, D, E, T)  # warm-up / JIT
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(dx, didx, dout, dcounts, T, D, E, T)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    print(f"Perf T={T} D={D} E={E}: {ms:.4f} ms, "
          f"{2 * T * D * 4 / (ms * 1e6):.1f} GB/s")
    print("ALL OK" if ok else "FAILURES")
