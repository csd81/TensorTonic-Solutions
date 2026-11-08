"""Numba CUDA port of reduction.cu.

Parallel sum reduction of a float32 array:
  * grid-stride accumulation per thread,
  * warp reduction with cuda.shfl_down_sync,
  * block reduction through a 32-element shared array,
  * one atomicAdd per block into output[0].

solve(input, output, N) keeps the challenge signature. Host NumPy arrays and
Numba device arrays are both accepted; output must contain one element.
"""

import numpy as np
from numba import cuda, float32


@cuda.jit
def reduce_kernel(input, output, N):
    shared = cuda.shared.array(32, float32)

    tid = cuda.threadIdx.x
    lane = tid & 31
    wid = tid >> 5

    total = float32(0.0)
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(idx, N, stride):
        total += input[i]

    for offset in (16, 8, 4, 2, 1):
        total += cuda.shfl_down_sync(0xFFFFFFFF, total, offset)

    if lane == 0:
        shared[wid] = total
    cuda.syncthreads()

    num_warps = cuda.blockDim.x // 32
    val = shared[lane] if lane < num_warps else float32(0.0)

    if wid == 0:
        for offset in (16, 8, 4, 2, 1):
            val += cuda.shfl_down_sync(0xFFFFFFFF, val, offset)
        if lane == 0:
            cuda.atomic.add(output, 0, val)


@cuda.jit
def zero_kernel(output):
    output[0] = 0.0


def _as_1d(array, n):
    if getattr(array, "ndim", 1) == 1:
        return array
    return array.reshape(n)


def solve(input, output, N):
    if N < 1:
        raise ValueError("N must be positive")

    in1 = _as_1d(input, N)
    out1 = _as_1d(output, 1)
    # A device-side zero avoids the synchronous host->device scalar copy
    # that output[0] = 0.0 would perform.
    zero_kernel[1, 1](out1)

    threads = 256
    blocks = min((N + threads - 1) // threads, 1024)
    reduce_kernel[blocks, threads](in1, out1, N)
    cuda.synchronize()


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    cases = [
        ("example 1", np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32), 36.0),
        ("example 2", np.array([-2.5, 1.5, -1.0, 2.0], dtype=np.float32), 0.0),
        ("single", np.array([7.5], dtype=np.float32), 7.5),
    ]
    rng = np.random.default_rng(42)
    for n in (1000, 1000003, 4194304):
        data = rng.uniform(-1000.0, 1000.0, n).astype(np.float32)
        cases.append((f"random N={n}", data, float(data.astype(np.float64).sum())))

    for name, data, expected in cases:
        d_in = cuda.to_device(data)
        d_out = cuda.device_array(1, dtype=np.float32)
        solve(d_in, d_out, data.size)
        got = float(d_out.copy_to_host()[0])
        sum_abs = float(np.abs(data.astype(np.float64)).sum())
        rel = abs(got - expected) / max(1.0, sum_abs)
        print(f"{name:>18}: got={got:.9g} expected={expected:.9g} "
              f"rel={rel:.3e} {'OK' if rel < 1e-5 else 'FAIL'}")

    # Host-array path
    h = rng.uniform(-1000.0, 1000.0, 1000).astype(np.float32)
    h_out = np.zeros(1, dtype=np.float32)
    solve(h, h_out, h.size)
    ref = float(h.astype(np.float64).sum())
    print(f"host-array path: got={h_out[0]:.9g} expected={ref:.9g} "
          f"{'OK' if abs(h_out[0] - ref) / max(1.0, np.abs(h).sum()) < 1e-5 else 'FAIL'}")

    # Performance: 4,194,304 ones, median of 20 runs.
    import statistics
    import time

    n = 4194304
    d_in = cuda.to_device(np.ones(n, dtype=np.float32))
    d_out = cuda.device_array(1, dtype=np.float32)
    solve(d_in, d_out, n)  # warm-up
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        solve(d_in, d_out, n)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    got = float(d_out.copy_to_host()[0])
    gbps = n * 4 / (ms * 1e6)
    print(f"Perf N={n}: {ms:.3f} ms, ~{gbps:.1f} GB/s, result={got:.1f} "
          f"({'OK' if got == float(n) else 'FAIL'})")
