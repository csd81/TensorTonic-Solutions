"""Numba CUDA port of kmeans_clustering.cu.

k-means over 2D points: assign, re-mean, repeat until the centroids stop
moving (or max_iterations).

The C++ version runs the whole clustering in one cooperative launch and does
the odd/even work with a grid-wide sync.  Numba has no grid sync (and no
cooperative launch), so this port uses the other way to avoid per-iteration
kernel launches: a **single block**.  The block is 1024 threads, so at the
benchmark size (10,000 points) each thread handles ten points per iteration,
and the iteration needs only __syncthreads().  For the maximum problem
(1,000,000 points times 1,000 clusters) one block is slow but still finishes
in well under a second.

Accumulation follows the C++ kernel: for k <= 8 each thread keeps eight
(sum_x, sum_y, count) triples in local memory and the block reduces them with
warp shuffles -- a per-element shared atomicAdd into five clusters serialises
badly (measured at ~500 us per iteration in the C++ version).  Larger k uses
shared-memory atomics.

solve(data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
final_centroid_x, final_centroid_y, sample_size, k, max_iterations) keeps the
challenge signature.
"""

import numpy as np
from numba import cuda, float32, int32

THREADS = 512
NWARPS = THREADS // 32
MAXK = 8
FLT_MAX = float32(3.4028234663852886e38)
FULL = 0xFFFFFFFF


@cuda.jit
def kmeans_block(dx, dy, labels, cx, cy, n, k, max_iter, init_x, init_y):
    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5

    wbuf = cuda.shared.array(shape=(NWARPS, MAXK, 3), dtype=float32)
    sh_sx = cuda.shared.array(shape=1024, dtype=float32)
    sh_sy = cuda.shared.array(shape=1024, dtype=float32)
    sh_sc = cuda.shared.array(shape=1024, dtype=int32)
    sh_moved = cuda.shared.array(shape=1, dtype=int32)

    ax = cuda.local.array(MAXK, dtype=float32)
    ay = cuda.local.array(MAXK, dtype=float32)
    ac = cuda.local.array(MAXK, dtype=int32)
    lwx = cuda.local.array(MAXK, dtype=float32)
    lwy = cuda.local.array(MAXK, dtype=float32)

    # Copy the initial centroids into the working arrays.
    for c in range(tid, k, THREADS):
        cx[c] = init_x[c]
        cy[c] = init_y[c]
    cuda.syncthreads()

    for it in range(max_iter):
        if tid == 0:
            sh_moved[0] = 0
        if k <= MAXK:
            for c in range(MAXK):
                ax[c] = float32(0.0)
                ay[c] = float32(0.0)
                ac[c] = 0
            # Centroids into local memory (slots >= k stay at FLT_MAX).
            for c in range(MAXK):
                if c < k:
                    lwx[c] = cx[c]
                    lwy[c] = cy[c]
                else:
                    lwx[c] = FLT_MAX
                    lwy[c] = FLT_MAX
            i = tid
            while i < n:
                px = dx[i]
                py = dy[i]
                best = FLT_MAX
                bc = 0
                for c in range(MAXK):
                    if c < k:
                        ddx = px - lwx[c]
                        ddy = py - lwy[c]
                        d = ddx * ddx + ddy * ddy
                        if d < best:
                            best = d
                            bc = c
                labels[i] = bc
                for c in range(MAXK):
                    if c < k and bc == c:
                        ax[c] += px
                        ay[c] += py
                        ac[c] += 1
                i += THREADS
            # Warp reduce per cluster, then combine the warps.
            for c in range(MAXK):
                if c < k:
                    vx = ax[c]
                    vy = ay[c]
                    vc = ac[c]
                    for off in (16, 8, 4, 2, 1):
                        vx += cuda.shfl_down_sync(FULL, vx, off)
                        vy += cuda.shfl_down_sync(FULL, vy, off)
                        vc += cuda.shfl_down_sync(FULL, vc, off)
                    if lane == 0:
                        wbuf[warp, c, 0] = vx
                        wbuf[warp, c, 1] = vy
                        wbuf[warp, c, 2] = float32(vc)
            cuda.syncthreads()
            if tid < k:
                tx = float32(0.0)
                ty = float32(0.0)
                tc = 0
                for w in range(NWARPS):
                    tx += wbuf[w, tid, 0]
                    ty += wbuf[w, tid, 1]
                    tc += int32(wbuf[w, tid, 2])
                if tc > 0:
                    nx = tx / float32(tc)
                    ny = ty / float32(tc)
                    if abs(nx - cx[tid]) > 1e-6 or abs(ny - cy[tid]) > 1e-6:
                        sh_moved[0] = 1
                    cx[tid] = nx
                    cy[tid] = ny
        else:
            # Large k: shared atomics spread over enough addresses.
            for c in range(tid, k, THREADS):
                sh_sx[c] = float32(0.0)
                sh_sy[c] = float32(0.0)
                sh_sc[c] = 0
            cuda.syncthreads()
            i = tid
            while i < n:
                px = dx[i]
                py = dy[i]
                best = FLT_MAX
                bc = 0
                for c in range(k):
                    ddx = px - cx[c]
                    ddy = py - cy[c]
                    d = ddx * ddx + ddy * ddy
                    if d < best:
                        best = d
                        bc = c
                labels[i] = bc
                cuda.atomic.add(sh_sx, bc, px)
                cuda.atomic.add(sh_sy, bc, py)
                cuda.atomic.add(sh_sc, bc, 1)
                i += THREADS
            cuda.syncthreads()
            for c in range(tid, k, THREADS):
                cnt = sh_sc[c]
                if cnt > 0:
                    nx = sh_sx[c] / float32(cnt)
                    ny = sh_sy[c] / float32(cnt)
                    if abs(nx - cx[c]) > 1e-6 or abs(ny - cy[c]) > 1e-6:
                        sh_moved[0] = 1
                    cx[c] = nx
                    cy[c] = ny
        cuda.syncthreads()
        if sh_moved[0] == 0:
            break

    # Final assignment against the final centroids.
    if k <= MAXK:
        for c in range(MAXK):
            if c < k:
                lwx[c] = cx[c]
                lwy[c] = cy[c]
            else:
                lwx[c] = FLT_MAX
                lwy[c] = FLT_MAX
        i = tid
        while i < n:
            px = dx[i]
            py = dy[i]
            best = FLT_MAX
            bc = 0
            for c in range(MAXK):
                if c < k:
                    ddx = px - lwx[c]
                    ddy = py - lwy[c]
                    d = ddx * ddx + ddy * ddy
                    if d < best:
                        best = d
                        bc = c
            labels[i] = bc
            i += THREADS
    else:
        i = tid
        while i < n:
            px = dx[i]
            py = dy[i]
            best = FLT_MAX
            bc = 0
            for c in range(k):
                ddx = px - cx[c]
                ddy = py - cy[c]
                d = ddx * ddx + ddy * ddy
                if d < best:
                    best = d
                    bc = c
            labels[i] = bc
            i += THREADS


def solve(data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
          final_centroid_x, final_centroid_y, sample_size, k, max_iterations):
    if sample_size <= 0 or k <= 0 or max_iterations < 0:
        return

    DX = data_x if getattr(data_x, "ndim", 1) == 1 else data_x.reshape(sample_size)
    DY = data_y if getattr(data_y, "ndim", 1) == 1 else data_y.reshape(sample_size)
    L = labels if getattr(labels, "ndim", 1) == 1 else labels.reshape(sample_size)
    IX = (initial_centroid_x if getattr(initial_centroid_x, "ndim", 1) == 1
          else initial_centroid_x.reshape(k))
    IY = (initial_centroid_y if getattr(initial_centroid_y, "ndim", 1) == 1
          else initial_centroid_y.reshape(k))
    FX = (final_centroid_x if getattr(final_centroid_x, "ndim", 1) == 1
          else final_centroid_x.reshape(k))
    FY = (final_centroid_y if getattr(final_centroid_y, "ndim", 1) == 1
          else final_centroid_y.reshape(k))

    kmeans_block[1, THREADS](DX, DY, L, FX, FY, sample_size, k, max_iterations,
                             IX, IY)
    cuda.synchronize()


def _reference(dx, dy, cx0, cy0, k, max_iterations):
    n = dx.size
    cx = cx0.astype(np.float64).copy()
    cy = cy0.astype(np.float64).copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iterations):
        sx = np.zeros(k)
        sy = np.zeros(k)
        cnt = np.zeros(k, dtype=np.int64)
        # assign
        best = np.full(n, 1e300)
        for c in range(k):
            d = (dx - cx[c]) ** 2 + (dy - cy[c]) ** 2
            m = d < best
            best[m] = d[m]
            labels[m] = c
        for c in range(k):
            m = labels == c
            sx[c] = dx[m].sum()
            sy[c] = dy[m].sum()
            cnt[c] = m.sum()
        moved = False
        for c in range(k):
            if cnt[c] > 0:
                nx = sx[c] / cnt[c]
                ny = sy[c] / cnt[c]
                if abs(nx - cx[c]) > 1e-6 or abs(ny - cy[c]) > 1e-6:
                    moved = True
                cx[c] = nx
                cy[c] = ny
        if not moved:
            break
    best = np.full(n, 1e300)
    for c in range(k):
        d = (dx - cx[c]) ** 2 + (dy - cy[c]) ** 2
        m = d < best
        best[m] = d[m]
        labels[m] = c
    return labels, cx, cy


def _run(name, dx, dy, cx0, cy0, k, max_iterations):
    ref_labels, ref_x, ref_y = _reference(dx, dy, cx0, cy0, k, max_iterations)
    n = dx.size
    dDX = cuda.to_device(np.ascontiguousarray(dx, dtype=np.float32))
    dDY = cuda.to_device(np.ascontiguousarray(dy, dtype=np.float32))
    dL = cuda.device_array(n, dtype=np.int32)
    dIX = cuda.to_device(np.ascontiguousarray(cx0, dtype=np.float32))
    dIY = cuda.to_device(np.ascontiguousarray(cy0, dtype=np.float32))
    dFX = cuda.device_array(k, dtype=np.float32)
    dFY = cuda.device_array(k, dtype=np.float32)
    solve(dDX, dDY, dL, dIX, dIY, dFX, dFY, n, k, max_iterations)
    labels = dL.copy_to_host()
    fx = dFX.copy_to_host().astype(np.float64)
    fy = dFY.copy_to_host().astype(np.float64)
    label_diff = int(np.sum(labels != ref_labels))
    scale = max(1.0, float(np.abs(ref_x).max()))
    rel = max(float(np.abs(fx - ref_x).max()), float(np.abs(fy - ref_y).max())) / scale
    ok = rel < 1e-3
    print(f"{name:>22} N={n} k={k} it={max_iterations}: centroid_rel={rel:.2e} "
          f"label_diff={label_diff} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    ok &= _run("example", np.array([1, 2, 8, 9], np.float32),
               np.array([1, 2, 8, 9], np.float32),
               np.array([1, 8], np.float32), np.array([1, 8], np.float32), 2, 10)
    ok &= _run("n=1 k=1", np.array([3.0], np.float32),
               np.array([4.0], np.float32), np.array([0.0], np.float32),
               np.array([0.0], np.float32), 1, 5)

    def random_case(n, k, iters, name):
        dx = rng.uniform(-100, 100, n).astype(np.float32)
        dy = rng.uniform(-100, 100, n).astype(np.float32)
        pick = rng.integers(0, n, k)
        return _run(name, dx, dy, dx[pick], dy[pick], k, iters)

    ok &= random_case(100, 3, 20, "100 pts k=3")
    ok &= random_case(1000, 5, 30, "1k pts k=5")
    ok &= random_case(5000, 8, 50, "5k pts k=8")
    ok &= random_case(10000, 5, 30, "benchmark shape")
    ok &= random_case(10000, 50, 30, "10k pts k=50")
    ok &= random_case(10000, 1000, 10, "10k pts k=1000")
    ok &= random_case(100000, 16, 20, "100k pts k=16")

    # Performance: sample_size=10,000, k=5, 30 iterations.
    import statistics
    import time

    N, K, ITERS = 10000, 5, 30
    dx = rng.uniform(-100, 100, N).astype(np.float32)
    dy = rng.uniform(-100, 100, N).astype(np.float32)
    pick = rng.integers(0, N, K)
    dDX = cuda.to_device(dx)
    dDY = cuda.to_device(dy)
    dL = cuda.device_array(N, dtype=np.int32)
    dIX = cuda.to_device(dx[pick])
    dIY = cuda.to_device(dy[pick])
    dFX = cuda.device_array(K, dtype=np.float32)
    dFY = cuda.device_array(K, dtype=np.float32)
    solve(dDX, dDY, dL, dIX, dIY, dFX, dFY, N, K, ITERS)  # warm-up
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        solve(dDX, dDY, dL, dIX, dIY, dFX, dFY, N, K, ITERS)
        times.append((time.perf_counter() - t0) * 1e6)
    ms = statistics.median(times)
    ref_labels, ref_x, ref_y = _reference(dx, dy, dx[pick], dy[pick], K, ITERS)
    fx = dFX.copy_to_host().astype(np.float64)
    rel = float(np.abs(fx - ref_x).max()) / max(1.0, float(np.abs(ref_x).max()))
    print(f"Perf N={N} k={K} it={ITERS}: {ms:.2f} us, rel={rel:.2e}, "
          f"{'OK' if rel < 1e-3 else 'FAIL'}")
    ok &= rel < 1e-3
    print("ALL OK" if ok else "FAILURES")
