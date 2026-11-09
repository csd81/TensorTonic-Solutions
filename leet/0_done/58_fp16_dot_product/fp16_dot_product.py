"""Numba CUDA port of fp16_dot_product.cu.

result = sum_i A[i] * B[i] for FP16 vectors, FP32 accumulation, half output.
N <= 1e8; benchmark N = 1e8 (400 MB of reads, so read-bound: ~87 GB/s ->
4.6 ms).  The C++ kernel measures 4.62 ms at 86.6 GB/s against the starter's
5.13 ms.

Numba cannot index `float16` arrays, so -- as in the GEMM port -- the buffers
are passed as `int16` bit-pattern views and converted with
`libdevice.half2float` / `libdevice.float2half_rn`.  The port also views the
same buffers as `int32` so each load fetches two halves (4 bytes per lane),
splits them with shifts and masks, and feeds two FP32 FMAs; the odd tail
element is handled by one thread through the int16 view.  Reduction is warp
shuffles + one atomic per block into a cached 1-element accumulator, then a
1-thread convert kernel.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, int32, int16
from numba.cuda import libdevice

THREADS = 256
BLOCKS_CAP = 1024

_sum = None


@cuda.jit(cache=True)
def dot_kernel(a32, b32, a16, b16, sum_ptr, n2, N):
    ws = cuda.shared.array(8, dtype=float32)  # one slot per warp (256 thr)
    s0 = float32(0.0)
    s1 = float32(0.0)

    tid = int32(cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x)
    stride = int32(cuda.blockDim.x * cuda.gridDim.x)

    i = tid
    while i < n2:
        va = a32[i]
        vb = b32[i]
        alo = libdevice.half2float(int16(va & 65535))
        ahi = libdevice.half2float(int16((va >> 16) & 65535))
        blo = libdevice.half2float(int16(vb & 65535))
        bhi = libdevice.half2float(int16((vb >> 16) & 65535))
        s0 += alo * blo
        s1 += ahi * bhi
        i += stride

    # Odd-length tail: the last element has no int32 partner.
    if cuda.blockIdx.x == 0 and cuda.threadIdx.x == 0 and (N & 1) == 1:
        s0 += (libdevice.half2float(a16[N - 1]) *
               libdevice.half2float(b16[N - 1]))

    # Warp reduction.
    s = s0 + s1
    off = int32(16)
    while off > 0:
        s += cuda.shfl_down_sync(0xFFFFFFFF, s, off)
        off >>= 1
    lane = int32(cuda.threadIdx.x) & 31
    warp = int32(cuda.threadIdx.x) >> 5
    if lane == 0:
        ws[warp] = s
    cuda.syncthreads()
    if cuda.threadIdx.x == 0:
        total = float32(0.0)
        w = int32(0)
        while w < (int32(cuda.blockDim.x) + int32(31)) >> 5:
            total += ws[w]
            w += int32(1)
        cuda.atomic.add(sum_ptr, 0, total)


@cuda.jit(cache=True)
def convert_kernel(sum_ptr, out):
    if cuda.grid(1) == 0:
        out[0] = libdevice.float2half_rn(sum_ptr[0])


def solve(a16, b16, out16, N):
    """int16 bit-pattern views of the half device arrays; out16 is int16[1]."""
    global _sum
    if N <= 0:
        return
    if _sum is None:
        _sum = cuda.device_array(1, dtype=np.float32)
    _sum[0] = np.float32(0.0)
    # An odd-length int16 buffer cannot be reinterpreted as int32 wholesale;
    # the trailing element is handled through the int16 view in the kernel.
    even = N - (N & 1)
    a32 = a16[:even].view(np.int32)
    b32 = b16[:even].view(np.int32)
    n2 = N // 2
    blocks = min(BLOCKS_CAP, max(1, (n2 + THREADS - 1) // THREADS))
    dot_kernel[blocks, THREADS](a32, b32, a16, b16, _sum, int32(n2), int32(N))
    convert_kernel[1, 1](_sum, out16)
    cuda.synchronize()


def _ref_dot(a, b):
    return float(np.sum(a.astype(np.float64) * b.astype(np.float64)))


def _check(name, a, b, N):
    ref = _ref_dot(a, b)
    da = cuda.to_device(np.ascontiguousarray(a, dtype=np.float16).view(np.int16))
    db = cuda.to_device(np.ascontiguousarray(b, dtype=np.float16).view(np.int16))
    dout = cuda.device_array(1, dtype=np.int16)
    solve(da, db, dout, N)
    got = float(np.float16(dout.copy_to_host()[0].view(np.float16)))
    # The result is stored as half: compare at half precision.
    tol = 1e-3 * abs(ref) + 1e-2
    ok = abs(got - ref) <= tol
    print(f"{name:>16} N={N}: got={got:.6g} ref={ref:.6g} err={abs(got-ref):.2e} "
          f"{'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    ok &= _check("example 1",
                 np.array([1, 2, 3, 4], np.float16),
                 np.array([5, 6, 7, 8], np.float16), 4)
    ok &= _check("example 2",
                 np.array([0.5, 1.5, 2.5], np.float16),
                 np.array([2, 3, 4], np.float16), 3)
    for N in (1, 7, 8, 1000, 100000, 1 << 20):
        a = rng.uniform(-1, 1, N).astype(np.float16)
        b = rng.uniform(-1, 1, N).astype(np.float16)
        ok &= _check(f"random N={N}", a, b, N)

    # Benchmark: N = 1e8.
    import statistics
    import time

    N = 100000000
    a = rng.uniform(-1, 1, N).astype(np.float16)
    b = rng.uniform(-1, 1, N).astype(np.float16)
    ok &= _check("benchmark shape", a, b, N)
    da = cuda.to_device(a.view(np.int16))
    db = cuda.to_device(b.view(np.int16))
    dout = cuda.device_array(1, dtype=np.int16)
    solve(da, db, dout, N)  # warm-up / JIT
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(da, db, dout, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    print(f"Perf N={N}: {ms:.4f} ms, {2 * N * 2 / (ms * 1e6):.1f} GB/s")
    print("ALL OK" if ok else "FAILURES")
