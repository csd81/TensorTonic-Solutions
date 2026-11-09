"""Numba CUDA port of hadamard_transform.cu.

Batched orthonormal Walsh-Hadamard transform: y[row] = (1/sqrt(N)) H_N x[row]
for B rows of length N (a power of two).  Benchmark: B=8192, N=4096
(268 MB of traffic; the C++ copy floor is 3.29 ms and the shipped kernel
reaches 3.41 ms).

The C++ kernel does the h = 1, 2 and 4 stages in registers while each thread
loads 8 consecutive floats (two float4s), then runs the remaining stages in
shared memory with float4 accesses and shift/mask index math, and lets the
last stage write straight to global memory.

Numba has neither float4 accesses nor a practical way to hold several
elements per lane without spilling, so this port is the classic shape: one
block per row, coalesced scalar loads into a dynamically sized shared array,
all log2(N) stages in shared memory, and coalesced scaled stores.  The
integer division/modulo per butterfly of the starter is replaced with
shifts and masks (h = 1<<e; section = v >> e, offset = v & (h-1)).

The shared array is dynamic: `cuda.shared.array(0, dtype)` with the byte size
passed at launch as the fourth launch-configuration item.  Measured 6.65-6.98 ms
at the benchmark shape (38-40 GB/s) with 256 threads per block; block sizes
128/256/512/1024 measure 8.20/6.98/7.75/9.38 ms.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, int32

THREADS = 256  # best of 128/256/512/1024 for this shape


@cuda.jit(cache=True)
def fwht_kernel(input, output, N, scale):
    s = cuda.shared.array(0, dtype=float32)
    row = int32(cuda.blockIdx.x)
    tid = int32(cuda.threadIdx.x)
    bd = int32(cuda.blockDim.x)
    base = row * N

    i = tid
    while i < N:
        s[i] = input[base + i]
        i += bd
    cuda.syncthreads()

    pairs = N >> 1
    e = int32(0)
    while (int32(1) << e) < N:
        h = int32(1) << e
        v = tid
        while v < pairs:
            section = v >> e
            off = v & (h - int32(1))
            ia = (section << (e + 1)) + off
            a = s[ia]
            b = s[ia + h]
            s[ia] = a + b
            s[ia + h] = a - b
            v += bd
        cuda.syncthreads()
        e += int32(1)

    i = tid
    while i < N:
        output[base + i] = s[i] * scale
        i += bd


def solve(input, output, B, N):
    """input, output: float32 device arrays of B*N elements."""
    if B <= 0 or N <= 0:
        return
    scale = float32(1.0 / np.sqrt(float(N)))
    threads = min(THREADS, N)
    fwht_kernel[B, threads, 0, N * 4](input, output, int32(N), scale)
    cuda.synchronize()


def _ref(a, B, N):
    x = a.astype(np.float64).reshape(B, N).copy()
    h = 1
    while h < N:
        x = x.reshape(B, N // (2 * h), 2, h)
        lo = x[:, :, 0, :].copy()
        hi = x[:, :, 1, :].copy()
        x[:, :, 0, :] = lo + hi
        x[:, :, 1, :] = lo - hi
        x = x.reshape(B, N)
        h *= 2
    return x / np.sqrt(float(N))


def _check(name, a, B, N):
    ref = _ref(a, B, N)
    din = cuda.to_device(np.ascontiguousarray(a, dtype=np.float32).reshape(-1))
    dout = cuda.device_array(B * N, dtype=np.float32)
    solve(din, dout, B, N)
    got = dout.copy_to_host().astype(np.float64).reshape(B, N)
    err = np.abs(got - ref)
    bad = int(np.count_nonzero(~(err <= 1e-4 + 1e-4 * np.abs(ref))))
    print(f"{name:>18} B={B} N={N}: bad={bad} max_abs={err.max():.2e} "
          f"{'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _check("example", np.array([1.0, 2.0, 3.0, 4.0], np.float32), 1, 4)
    for B, N in ((1, 1), (1, 2), (3, 8), (17, 64), (64, 256), (129, 512),
                 (1000, 1024), (8192, 4096)):
        a = rng.uniform(-1, 1, B * N).astype(np.float32)
        ok &= _check(f"B={B} N={N}", a, B, N)

    # Benchmark: B = 8192, N = 4096.
    import statistics
    import time

    B, N = 8192, 4096
    a = rng.uniform(-1, 1, B * N).astype(np.float32)
    din = cuda.to_device(a)
    dout = cuda.device_array(B * N, dtype=np.float32)
    solve(din, dout, B, N)  # warm-up / JIT
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(din, dout, B, N)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    print(f"Perf B={B} N={N}: {ms:.4f} ms, "
          f"{2 * B * N * 4 / (ms * 1e6):.1f} GB/s")
    print("ALL OK" if ok else "FAILURES")
