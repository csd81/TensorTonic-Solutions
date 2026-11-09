"""Numba CUDA port of rms_norm_backward.cu.

RMS-norm backward: rstd_i from x, dx = rstd*(dy*w - xhat*mean(dy*w*xhat)),
dweight_j = sum_i dy*xhat.  Benchmark N=16,384, C=4,096 (x, dy, dx are 268 MB
each).  The C++ kernel measures 10.5 ms (77 GB/s of the 804 MB minimum)
against the starter's 21.7 ms, which reads x three times and dy twice and
takes one float atomic per element for dweight.

The port keeps the C++ kernel's accuracy strategy without its register
blocking: R rows per block accumulate dweight's contribution for the block's
columns in a shared array (each column is owned by one thread for the whole
block, so the updates are race-free), and only then does one float64 atomic
per column per block -- 4.2 K doubles per column instead of 16 K float adds,
which removes the starter's ~1e-2 cancellation error on dweight.  The row
data itself is re-read from L1 for the normalize pass (Numba cannot hold 16
columns in registers across a reduction without spilling to local memory).
dweight accumulates in a cached float64 buffer and is converted to float32 by
a final kernel.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, float64, int32
from numba.cuda import libdevice

TPB = 256
R = 8
MAX_C_SHARED = 4096


@cuda.jit(cache=True)
def zero_kernel(a, n):
    i = int32(cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x)
    if i < n:
        a[i] = 0.0


@cuda.jit(cache=True)
def rms_bwd_kernel(x, weight, dy, dx, dweight64, N, C, eps):
    scratch = cuda.shared.array(8, dtype=float32)
    dws = cuda.shared.array(MAX_C_SHARED, dtype=float32)
    j = int32(cuda.threadIdx.x)
    nwarp = (int32(cuda.blockDim.x) + int32(31)) >> 5

    # Zero this block's dweight accumulation (each column is owned by one
    # thread for the whole block).
    c = j
    while c < C:
        dws[c] = float32(0.0)
        c += int32(TPB)
    cuda.syncthreads()

    row0 = int32(cuda.blockIdx.x) * int32(R)
    r = int32(0)
    while r < int32(R):
        row = row0 + r
        if row >= N:
            break
        ss = float32(0.0)
        c = j
        while c < C:
            v = x[row * C + c]
            ss += v * v
            c += int32(TPB)
        ss = broadcast_reduce(ss, scratch, nwarp)
        rstd = libdevice.rsqrtf(ss / float32(C) + eps)

        dot = float32(0.0)
        c = j
        while c < C:
            xh = x[row * C + c] * rstd
            dot += (dy[row * C + c] * weight[c]) * xh
            c += int32(TPB)
        mean = broadcast_reduce(dot, scratch, nwarp) / float32(C)

        c = j
        while c < C:
            v = x[row * C + c]
            xh = v * rstd
            dx[row * C + c] = rstd * (dy[row * C + c] * weight[c] - xh * mean)
            dws[c] += dy[row * C + c] * xh
            c += int32(TPB)
        r += int32(1)

    cuda.syncthreads()
    c = j
    while c < C:
        if dws[c] != float32(0.0):
            cuda.atomic.add(dweight64, c, float64(dws[c]))
        c += int32(TPB)

@cuda.jit(device=True)
def broadcast_reduce(val, scratch, nwarp):
    v = val
    off = int32(16)
    while off > 0:
        v += cuda.shfl_down_sync(0xFFFFFFFF, v, off)
        off >>= 1
    lane = int32(cuda.threadIdx.x) & 31
    warp = int32(cuda.threadIdx.x) >> 5
    if lane == 0:
        scratch[warp] = v
    cuda.syncthreads()
    if cuda.threadIdx.x == 0:
        t = float32(0.0)
        k = int32(0)
        while k < nwarp:
            t += scratch[k]
            k += int32(1)
        scratch[0] = t
    cuda.syncthreads()
    r = scratch[0]
    cuda.syncthreads()  # the next reduction reuses scratch[0]
    return r


@cuda.jit(cache=True)
def convert_kernel(src, dst, C):
    j = int32(cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x)
    if j < C:
        dst[j] = float32(src[j])


_dw64 = None


def solve(x, weight, dy, dx, dweight, N, C, eps):
    """dweight is a float32 device array; accumulation happens in float64."""
    global _dw64
    if N <= 0 or C <= 0:
        return
    if _dw64 is None or _dw64.size < C:
        _dw64 = cuda.device_array(C, dtype=np.float64)
    dw64 = _dw64[:C]
    zero_kernel[(C + 255) // 256, 256](dw64, int32(C))
    blocks = (N + R - 1) // R
    rms_bwd_kernel[blocks, TPB](x, weight, dy, dx, dw64, int32(N), int32(C),
                                float32(eps))
    convert_kernel[(C + 255) // 256, 256](dw64, dweight, int32(C))
    cuda.synchronize()

def _ref(x, w, dy, N, C, eps):
    x64 = x.reshape(N, C).astype(np.float64)
    dy64 = dy.reshape(N, C).astype(np.float64)
    w64 = w.astype(np.float64)
    rstd = 1.0 / np.sqrt((x64 * x64).mean(axis=1, keepdims=True) + eps)
    xh = x64 * rstd
    g = dy64 * w64
    mean = (g * xh).mean(axis=1, keepdims=True)
    dx = rstd * (g - xh * mean)
    dw = (dy64 * xh).sum(axis=0)
    return dx.reshape(-1), dw


def _check(name, x, w, dy, N, C, eps=1e-5):
    rdx, rdw = _ref(x, w, dy, N, C, eps)
    dxp = cuda.to_device(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))
    dwp = cuda.to_device(np.ascontiguousarray(w, dtype=np.float32))
    dyp = cuda.to_device(np.ascontiguousarray(dy, dtype=np.float32).reshape(-1))
    dxo = cuda.device_array(N * C, dtype=np.float32)
    dwo = cuda.device_array(C, dtype=np.float32)
    solve(dxp, dwp, dyp, dxo, dwo, N, C, eps)
    got_dx = dxo.copy_to_host().astype(np.float64)
    got_dw = dwo.copy_to_host().astype(np.float64)
    e_dx = np.abs(got_dx - rdx)
    e_dw = np.abs(got_dw - rdw)
    bad = int(np.count_nonzero(~(e_dx <= 1e-4 + 1e-4 * np.abs(rdx))))
    bad += int(np.count_nonzero(~(e_dw <= 1e-3 + 1e-4 * np.abs(rdw))))
    print(f"{name:>16} N={N} C={C}: bad={bad} max_dx={e_dx.max():.2e} "
          f"max_dw={e_dw.max():.2e} {'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _check("example", np.array([1, -1, 1, -1, 2, 0, 0, 0], np.float32),
                 np.array([1, 2, 0.5, 1], np.float32),
                 np.array([1, 1, 1, 1, 1, 0, 0, 1], np.float32), 2, 4)
    for N, C in ((1, 1), (1, 7), (5, 3), (17, 33), (1000, 64), (3, 1024),
                 (2, 4096)):
        x = rng.uniform(-250, 250, N * C).astype(np.float32)
        dy = rng.uniform(-10, 10, N * C).astype(np.float32)
        w = rng.uniform(-2, 2, C).astype(np.float32)
        ok &= _check(f"{N}x{C}", x, w, dy, N, C)

    # Benchmark: N=16384, C=4096.
    import statistics
    import time

    N, C = 16384, 4096
    x = rng.uniform(-250, 250, N * C).astype(np.float32)
    dy = rng.uniform(-10, 10, N * C).astype(np.float32)
    w = rng.uniform(-2, 2, C).astype(np.float32)
    ok &= _check("benchmark shape", x, w, dy, N, C)
    dxp = cuda.to_device(x)
    dwp = cuda.to_device(w)
    dyp = cuda.to_device(dy)
    dxo = cuda.device_array(N * C, dtype=np.float32)
    dwo = cuda.device_array(C, dtype=np.float32)
    solve(dxp, dwp, dyp, dxo, dwo, N, C, 1e-5)  # warm-up / JIT
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        solve(dxp, dwp, dyp, dxo, dwo, N, C, 1e-5)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    print(f"Perf N={N} C={C}: {ms:.4f} ms, "
          f"{3 * N * C * 4 / (ms * 1e6):.1f} GB/s")
    print("ALL OK" if ok else "FAILURES")
