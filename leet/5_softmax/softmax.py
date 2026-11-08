"""Numba CUDA port of softmax.cu.

Stable softmax using the max trick, split into the same five kernels:
  1. per-block max (grid-stride, warp shuffle + shared reduction),
  2. final max over block maxima,
  3. exp(x - max) written to output + per-block sums,
  4. final sum over block sums,
  5. normalize output by the total sum.

solve(input, output, N) keeps the challenge signature. Input/output may be
2-D or flat; host arrays are copied by Numba (device arrays are faster).
"""

import math

import numpy as np
from numba import cuda, float32


@cuda.jit(device=True)
def _warp_max(val):
    for off in (16, 8, 4, 2, 1):
        other = cuda.shfl_down_sync(0xFFFFFFFF, val, off)
        if other > val:
            val = other
    return val


@cuda.jit(device=True)
def _warp_sum(val):
    for off in (16, 8, 4, 2, 1):
        val += cuda.shfl_down_sync(0xFFFFFFFF, val, off)
    return val


@cuda.jit
def reduce_max_kernel(input, block_maxes, N):
    shared = cuda.shared.array(32, float32)
    tid = cuda.threadIdx.x
    lane = tid & 31
    wid = tid >> 5

    val = -math.inf
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    for idx in range(i, N, stride):
        x = input[idx]
        if x > val:
            val = x

    val = _warp_max(val)
    if lane == 0:
        shared[wid] = val
    cuda.syncthreads()

    num_warps = cuda.blockDim.x // 32
    val = shared[lane] if lane < num_warps else -math.inf
    if wid == 0:
        val = _warp_max(val)
        if lane == 0:
            block_maxes[cuda.blockIdx.x] = val


@cuda.jit
def final_reduce_max_kernel(block_maxes, count):
    shared = cuda.shared.array(32, float32)
    tid = cuda.threadIdx.x
    lane = tid & 31
    wid = tid >> 5

    val = -math.inf
    for idx in range(tid, count, cuda.blockDim.x):
        x = block_maxes[idx]
        if x > val:
            val = x

    val = _warp_max(val)
    if lane == 0:
        shared[wid] = val
    cuda.syncthreads()

    num_warps = cuda.blockDim.x // 32
    val = shared[lane] if lane < num_warps else -math.inf
    if wid == 0:
        val = _warp_max(val)
        if lane == 0:
            block_maxes[0] = val


@cuda.jit
def exp_and_sum_kernel(input, output, d_max, block_sums, N):
    shared = cuda.shared.array(32, float32)
    tid = cuda.threadIdx.x
    lane = tid & 31
    wid = tid >> 5

    max_val = d_max[0]
    total = float32(0.0)
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    for idx in range(i, N, stride):
        val = math.exp(input[idx] - max_val)
        output[idx] = val
        total += val

    total = _warp_sum(total)
    if lane == 0:
        shared[wid] = total
    cuda.syncthreads()

    num_warps = cuda.blockDim.x // 32
    total = shared[lane] if lane < num_warps else float32(0.0)
    if wid == 0:
        total = _warp_sum(total)
        if lane == 0:
            block_sums[cuda.blockIdx.x] = total


@cuda.jit
def final_reduce_sum_kernel(block_sums, count):
    shared = cuda.shared.array(32, float32)
    tid = cuda.threadIdx.x
    lane = tid & 31
    wid = tid >> 5

    total = float32(0.0)
    for idx in range(tid, count, cuda.blockDim.x):
        total += block_sums[idx]

    total = _warp_sum(total)
    if lane == 0:
        shared[wid] = total
    cuda.syncthreads()

    num_warps = cuda.blockDim.x // 32
    total = shared[lane] if lane < num_warps else float32(0.0)
    if wid == 0:
        total = _warp_sum(total)
        if lane == 0:
            block_sums[0] = total


@cuda.jit
def normalize_kernel(output, d_sums, N):
    inv_sum = 1.0 / d_sums[0]
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    for idx in range(i, N, stride):
        output[idx] *= inv_sum


def _as_1d(array, n):
    if getattr(array, "ndim", 1) == 1:
        return array
    return array.reshape(n)


def solve(input, output, N):
    if N < 1:
        raise ValueError("N must be positive")

    in1 = _as_1d(input, N)
    out1 = _as_1d(output, N)

    threads = 256
    blocks = min((N + threads - 1) // threads, 1024)
    d_max = cuda.device_array(blocks, dtype=np.float32)
    d_sums = cuda.device_array(blocks, dtype=np.float32)

    reduce_max_kernel[blocks, threads](in1, d_max, N)
    final_reduce_max_kernel[1, threads](d_max, blocks)
    exp_and_sum_kernel[blocks, threads](in1, out1, d_max, d_sums, N)
    final_reduce_sum_kernel[1, threads](d_sums, blocks)
    normalize_kernel[blocks, threads](out1, d_sums, N)
    cuda.synchronize()


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    def run_case(name, data):
        ref = np.exp(data.astype(np.float64) - np.max(data))
        ref /= ref.sum()

        d_in = cuda.to_device(data)
        d_out = cuda.device_array(data.size, dtype=np.float32)
        solve(d_in, d_out, data.size)
        got = d_out.copy_to_host()

        max_err = float(np.max(np.abs(got - ref)))
        rel = max_err / max(1e-30, float(np.max(np.abs(ref))))
        print(f"{name:>18} N={data.size}: max_err={max_err:.3e} "
              f"rel={rel:.3e} sum={got.sum():.9f} "
              f"{'OK' if rel < 1e-5 else 'FAIL'}")

    run_case("example 1", np.array([1.0, 2.0, 3.0], dtype=np.float32))
    run_case("example 2",
             np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32))
    run_case("single element", np.array([5.0], dtype=np.float32))

    rng = np.random.default_rng(42)
    for n in (1000, 100000, 500000):
        run_case(f"random", rng.normal(0.0, 10.0, n).astype(np.float32))

    # Host-array path
    h = rng.normal(0.0, 10.0, 1000).astype(np.float32)
    h_out = np.zeros(1000, dtype=np.float32)
    solve(h, h_out, h.size)
    ref = np.exp(h.astype(np.float64) - h.max())
    ref /= ref.sum()
    print(f"host-array path: max_err={np.max(np.abs(h_out - ref)):.3e}")

    # Performance: 500,000 elements, median of 20 runs.
    import statistics
    import time

    n = 500000
    data = rng.normal(0.0, 10.0, n).astype(np.float32)
    d_in = cuda.to_device(data)
    d_out = cuda.device_array(n, dtype=np.float32)
    solve(d_in, d_out, n)  # warm-up
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        solve(d_in, d_out, n)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    out = d_out.copy_to_host()
    gbps = 5.0 * n * 4 / (ms * 1e6)
    print(f"Perf N={n}: {ms:.3f} ms, ~{gbps:.1f} GB/s, "
          f"sum={out.sum():.9f}")
