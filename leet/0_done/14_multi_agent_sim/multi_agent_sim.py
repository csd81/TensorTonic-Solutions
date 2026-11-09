"""Numba CUDA port of multi_agent_sim.cu.

Boids neighbourhood step: average the velocities of the agents within
r = 5 and move each agent by alpha = 0.05 of the difference.

Same two ideas as the C++ version:

  1. The spatial hash.  Agents are binned into cells of side r (hashed into a
     fixed bucket array, so no bounding box is needed), then each agent only
     tests the agents in its own cell and the eight neighbours -- an agent
     within r must be in one of them because a cell is exactly r wide.  At
     N = 10,000 that turns 10^8 distance tests into a few thousand.
  2. Cheaper comparisons: positions and velocities live in separate arrays so
     the inner loop reads 8 bytes per candidate and only touches the velocity
     when the radius test passes; every agent counts itself (distance 0) and
     subtracts once at the end, which removes the per-candidate self test.

The port uses a single-block scan of the bucket counts (the C++ version
splits it into two levels, which matters less here than the per-launch
overhead).

solve(agents, agents_next, N) keeps the challenge signature; 4 floats per
agent, [x, y, vx, vy].
"""

import math

import numpy as np
from numba import cuda, float32, int32

BS = 256
R2 = float32(25.0)
INV_R = float32(0.2)

_buckets = {}
_ps = None
_vs = None
_cap_n = 0
_cap_nb = 0


@cuda.jit
def zero_kernel(a, n):
    i = cuda.grid(1)
    if i < n:
        a[i] = 0


@cuda.jit(device=True)
def cell_hash(cx, cy, mask):
    return ((cx * 73856093) ^ (cy * 19349663)) & mask


@cuda.jit
def count_kernel(ag, n, mask, counts):
    i = cuda.grid(1)
    if i < n:
        x = ag[4 * i]
        y = ag[4 * i + 1]
        cx = int32(math.floor(x * INV_R))
        cy = int32(math.floor(y * INV_R))
        cuda.atomic.add(counts, cell_hash(cx, cy, mask), 1)


@cuda.jit
def scan_kernel(counts, starts, cursor, nb):
    wsum = cuda.shared.array(shape=8, dtype=int32)
    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    run = (nb + cuda.blockDim.x - 1) // cuda.blockDim.x
    base = tid * run

    s = int32(0)
    for j in range(run):
        if base + j < nb:
            s += counts[base + j]
    x = s
    for off in (1, 2, 4, 8, 16):
        y = cuda.shfl_up_sync(0xFFFFFFFF, x, off)
        if lane >= off:
            x += y
    if lane == 31:
        wsum[warp] = x
    cuda.syncthreads()
    if warp == 0:
        w = wsum[lane] if lane < 8 else int32(0)
        for off in (1, 2, 4, 8, 16):
            y = cuda.shfl_up_sync(0xFFFFFFFF, w, off)
            if lane >= off:
                w += y
        if lane < 8:
            wsum[lane] = w
    cuda.syncthreads()
    wbase = wsum[warp - 1] if warp > 0 else int32(0)
    acc = wbase + (x - s)
    for j in range(run):
        if base + j < nb:
            starts[base + j] = acc
            cursor[base + j] = acc
            acc += counts[base + j]


@cuda.jit
def scatter_kernel(ag, n, mask, cursor, ps, vs):
    i = cuda.grid(1)
    if i < n:
        x = ag[4 * i]
        y = ag[4 * i + 1]
        cx = int32(math.floor(x * INV_R))
        cy = int32(math.floor(y * INV_R))
        p = cuda.atomic.add(cursor, cell_hash(cx, cy, mask), 1)
        ps[2 * p] = x
        ps[2 * p + 1] = y
        vs[2 * p] = ag[4 * i + 2]
        vs[2 * p + 1] = ag[4 * i + 3]


@cuda.jit
def query_kernel(ag, out, starts, counts, ps, vs, n, mask):
    i = cuda.grid(1)
    if i < n:
        x = ag[4 * i]
        y = ag[4 * i + 1]
        vx = ag[4 * i + 2]
        vy = ag[4 * i + 3]
        cx = int32(math.floor(x * INV_R))
        cy = int32(math.floor(y * INV_R))

        hs = cuda.local.array(9, dtype=int32)
        nh = 0
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                h = cell_hash(cx + ox, cy + oy, mask)
                dup = 0
                for q in range(nh):
                    if hs[q] == h:
                        dup = 1
                if dup == 0:
                    hs[nh] = h
                    nh += 1

        sx = float32(0.0)
        sy = float32(0.0)
        cnt = 0
        for q in range(nh):
            s = starts[hs[q]]
            e = s + counts[hs[q]]
            for p in range(s, e):
                dx = x - ps[2 * p]
                dy = y - ps[2 * p + 1]
                if dx * dx + dy * dy < R2:
                    sx += vs[2 * p]
                    sy += vs[2 * p + 1]
                    cnt += 1

        sx -= vx  # self
        sy -= vy
        cnt -= 1
        avx = vx
        avy = vy
        if cnt > 0:
            avx = sx / float32(cnt)
            avy = sy / float32(cnt)
        nvx = vx + float32(0.05) * (avx - vx)
        nvy = vy + float32(0.05) * (avy - vy)
        out[4 * i] = x + nvx
        out[4 * i + 1] = y + nvy
        out[4 * i + 2] = nvx
        out[4 * i + 3] = nvy


def _reserve(nb, n):
    global _buckets, _ps, _vs, _cap_n, _cap_nb
    if nb > _cap_nb:
        _buckets = {
            "counts": cuda.device_array(nb, dtype=np.int32),
            "starts": cuda.device_array(nb, dtype=np.int32),
            "cursor": cuda.device_array(nb, dtype=np.int32),
        }
        _cap_nb = nb
    if n > _cap_n:
        _ps = cuda.device_array(2 * n, dtype=np.float32)
        _vs = cuda.device_array(2 * n, dtype=np.float32)
        _cap_n = n


def solve(agents, agents_next, N):
    if agents is None or agents_next is None:
        return
    if N <= 0:
        raise ValueError("N must be positive")

    A = agents if getattr(agents, "ndim", 1) == 1 else agents.reshape(4 * N)
    O = (agents_next if getattr(agents_next, "ndim", 1) == 1
         else agents_next.reshape(4 * N))

    nb = 1024
    while nb < N and nb < (1 << 20):
        nb <<= 1
    _reserve(nb, N)
    counts = _buckets["counts"]
    starts = _buckets["starts"]
    cursor = _buckets["cursor"]
    ps = _ps[: 2 * N]
    vs = _vs[: 2 * N]

    blocks = (N + BS - 1) // BS
    mask = nb - 1

    zero_kernel[(nb + BS - 1) // BS, BS](counts, nb)
    count_kernel[blocks, BS](A, N, mask, counts)
    scan_kernel[1, 256](counts, starts, cursor, nb)
    scatter_kernel[blocks, BS](A, N, mask, cursor, ps, vs)
    query_kernel[blocks, BS](A, O, starts, counts, ps, vs, N, mask)
    cuda.synchronize()


# ------------------------------------------------------------------- tests

def _reference(a, n):
    out = np.zeros(4 * n, dtype=np.float64)
    x = a[:, 0].astype(np.float64)
    y = a[:, 1].astype(np.float64)
    vx = a[:, 2].astype(np.float64)
    vy = a[:, 3].astype(np.float64)
    for i in range(n):
        dx = x[i] - x
        dy = y[i] - y
        m = (dx * dx + dy * dy) < 25.0
        m[i] = False
        cnt = int(m.sum())
        if cnt > 0:
            avx = vx[m].mean()
            avy = vy[m].mean()
        else:
            avx, avy = vx[i], vy[i]
        nvx = vx[i] + 0.05 * (avx - vx[i])
        nvy = vy[i] + 0.05 * (avy - vy[i])
        out[4 * i] = x[i] + nvx
        out[4 * i + 1] = y[i] + nvy
        out[4 * i + 2] = nvx
        out[4 * i + 3] = nvy
    return out


def _run(name, agents):
    a = agents.reshape(-1, 4).astype(np.float32)
    n = a.shape[0]
    ref = _reference(a, n)
    dA = cuda.to_device(np.ascontiguousarray(a.reshape(-1), dtype=np.float32))
    dO = cuda.device_array(4 * n, dtype=np.float32)
    solve(dA, dO, n)
    got = dO.copy_to_host().astype(np.float64)
    err = float(np.abs(got - ref).max()) if n else 0.0
    scale = max(1.0, float(np.abs(ref).max()))
    ok = err / scale < 1e-5
    print(f"{name:>22} N={n}: max_err={err:.3e} rel={err / scale:.2e} "
          f"{'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    ok &= _run("example", np.array([0.0, 0.0, 1.0, 0.0,
                                    3.0, 4.0, 0.0, -1.0], np.float32))
    ok &= _run("single", np.array([1.0, 2.0, 0.5, -0.5], np.float32))
    ok &= _run("3 close", np.array([0, 0, 1, 0, 1, 0, 0, 1,
                                    2, 0, -1, 0], np.float32))

    def random_case(n, spread, name):
        a = np.empty((n, 4), np.float32)
        a[:, 0:2] = rng.uniform(-spread, spread, (n, 2))
        a[:, 2:4] = rng.uniform(-1, 1, (n, 2))
        return _run(name, a.reshape(-1))

    ok &= random_case(100, 20.0, "100 dense")
    ok &= random_case(1000, 100.0, "1k spread 100")
    ok &= random_case(2000, 1000.0, "2k spread 1000")
    ok &= random_case(4000, 50.0, "4k dense")

    # Performance: N = 10,000.
    import statistics
    import time

    N = 10000
    for spread in (50.0, 200.0, 2000.0):
        a = np.empty((N, 4), np.float32)
        a[:, 0:2] = rng.uniform(-spread, spread, (N, 2))
        a[:, 2:4] = rng.uniform(-1, 1, (N, 2))
        dA = cuda.to_device(a.reshape(-1))
        dO = cuda.device_array(4 * N, dtype=np.float32)
        solve(dA, dO, N)  # warm-up
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            solve(dA, dO, N)
            times.append((time.perf_counter() - t0) * 1e6)
        us = statistics.median(times)
        ref = _reference(a, N)
        got = dO.copy_to_host().astype(np.float64)
        rel = float(np.abs(got - ref).max()) / max(1.0, float(np.abs(ref).max()))
        print(f"Perf N={N} spread={spread:.0f}: {us:.1f} us, rel={rel:.2e}, "
              f"{'OK' if rel < 1e-5 else 'FAIL'}")
        ok &= rel < 1e-5
    print("ALL OK" if ok else "FAILURES")
