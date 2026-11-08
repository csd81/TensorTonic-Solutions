"""Numba CUDA port of convolution_1d.cu.

1D "valid" convolution:
    out[i] = sum_{j < kernel_size} in[i + j] * k[j],  i < in_size - k_size + 1.

The CUDA C++ version keeps a sliding eight-element window in registers and
uses float4 shared-memory loads (four taps per 16-byte load).  Numba has no
vector shared loads, so this port uses the strided mapping instead: thread t
computes the output indices t, t+THREADS, ..., t+7*THREADS, which keeps
every shared-memory access conflict free and lets one broadcast of k[j] feed
8 FMAs.  The tap loop is manually unrolled by 8 to amortise loop overhead
(Numba does not unroll it on its own).  The kernel and the input tile live in
shared memory, as in the C++ version.

solve(input, kernel, output, input_size, kernel_size) keeps the challenge
signature.  Host arrays are copied to the device and back by Numba.
"""

import numpy as np
from numba import cuda, float32

MAX_KERNEL_SIZE = 2048
THREADS = 768
OUT_PER_THREAD = 8
TAPS_PER_ITER = 8
BLOCK_OUT = THREADS * OUT_PER_THREAD          # 6144 outputs per block
TILE_SIZE = BLOCK_OUT + MAX_KERNEL_SIZE + 2   # input tile with right halo


@cuda.jit
def conv1d_kernel(input, kernel, output, input_size, kernel_size):
    s_kernel = cuda.shared.array(shape=MAX_KERNEL_SIZE, dtype=float32)
    s_input = cuda.shared.array(shape=TILE_SIZE, dtype=float32)

    tid = cuda.threadIdx.x
    block_start = cuda.blockIdx.x * BLOCK_OUT
    output_size = input_size - kernel_size + 1

    # Stage the kernel.
    i = tid
    while i < kernel_size:
        s_kernel[i] = kernel[i]
        i += THREADS

    # Stage the input tile plus the right halo (zero beyond the input).
    tile_len = BLOCK_OUT + kernel_size - 1
    i = tid
    while i < tile_len:
        g = block_start + i
        if g < input_size:
            s_input[i] = input[g]
        else:
            s_input[i] = 0.0
        i += THREADS
    cuda.syncthreads()

    # 8 outputs per thread, strided by THREADS so that s_input accesses are
    # conflict free across the warp.
    base = tid

    a0 = float32(0.0)
    a1 = float32(0.0)
    a2 = float32(0.0)
    a3 = float32(0.0)
    a4 = float32(0.0)
    a5 = float32(0.0)
    a6 = float32(0.0)
    a7 = float32(0.0)

    j = 0
    while j + 8 <= kernel_size:
        k0 = s_kernel[j + 0]
        k1 = s_kernel[j + 1]
        k2 = s_kernel[j + 2]
        k3 = s_kernel[j + 3]
        k4 = s_kernel[j + 4]
        k5 = s_kernel[j + 5]
        k6 = s_kernel[j + 6]
        k7 = s_kernel[j + 7]
        a0 += s_input[base + 0 + j + 0] * k0
        a1 += s_input[base + 768 + j + 0] * k0
        a2 += s_input[base + 1536 + j + 0] * k0
        a3 += s_input[base + 2304 + j + 0] * k0
        a4 += s_input[base + 3072 + j + 0] * k0
        a5 += s_input[base + 3840 + j + 0] * k0
        a6 += s_input[base + 4608 + j + 0] * k0
        a7 += s_input[base + 5376 + j + 0] * k0
        a0 += s_input[base + 0 + j + 1] * k1
        a1 += s_input[base + 768 + j + 1] * k1
        a2 += s_input[base + 1536 + j + 1] * k1
        a3 += s_input[base + 2304 + j + 1] * k1
        a4 += s_input[base + 3072 + j + 1] * k1
        a5 += s_input[base + 3840 + j + 1] * k1
        a6 += s_input[base + 4608 + j + 1] * k1
        a7 += s_input[base + 5376 + j + 1] * k1
        a0 += s_input[base + 0 + j + 2] * k2
        a1 += s_input[base + 768 + j + 2] * k2
        a2 += s_input[base + 1536 + j + 2] * k2
        a3 += s_input[base + 2304 + j + 2] * k2
        a4 += s_input[base + 3072 + j + 2] * k2
        a5 += s_input[base + 3840 + j + 2] * k2
        a6 += s_input[base + 4608 + j + 2] * k2
        a7 += s_input[base + 5376 + j + 2] * k2
        a0 += s_input[base + 0 + j + 3] * k3
        a1 += s_input[base + 768 + j + 3] * k3
        a2 += s_input[base + 1536 + j + 3] * k3
        a3 += s_input[base + 2304 + j + 3] * k3
        a4 += s_input[base + 3072 + j + 3] * k3
        a5 += s_input[base + 3840 + j + 3] * k3
        a6 += s_input[base + 4608 + j + 3] * k3
        a7 += s_input[base + 5376 + j + 3] * k3
        a0 += s_input[base + 0 + j + 4] * k4
        a1 += s_input[base + 768 + j + 4] * k4
        a2 += s_input[base + 1536 + j + 4] * k4
        a3 += s_input[base + 2304 + j + 4] * k4
        a4 += s_input[base + 3072 + j + 4] * k4
        a5 += s_input[base + 3840 + j + 4] * k4
        a6 += s_input[base + 4608 + j + 4] * k4
        a7 += s_input[base + 5376 + j + 4] * k4
        a0 += s_input[base + 0 + j + 5] * k5
        a1 += s_input[base + 768 + j + 5] * k5
        a2 += s_input[base + 1536 + j + 5] * k5
        a3 += s_input[base + 2304 + j + 5] * k5
        a4 += s_input[base + 3072 + j + 5] * k5
        a5 += s_input[base + 3840 + j + 5] * k5
        a6 += s_input[base + 4608 + j + 5] * k5
        a7 += s_input[base + 5376 + j + 5] * k5
        a0 += s_input[base + 0 + j + 6] * k6
        a1 += s_input[base + 768 + j + 6] * k6
        a2 += s_input[base + 1536 + j + 6] * k6
        a3 += s_input[base + 2304 + j + 6] * k6
        a4 += s_input[base + 3072 + j + 6] * k6
        a5 += s_input[base + 3840 + j + 6] * k6
        a6 += s_input[base + 4608 + j + 6] * k6
        a7 += s_input[base + 5376 + j + 6] * k6
        a0 += s_input[base + 0 + j + 7] * k7
        a1 += s_input[base + 768 + j + 7] * k7
        a2 += s_input[base + 1536 + j + 7] * k7
        a3 += s_input[base + 2304 + j + 7] * k7
        a4 += s_input[base + 3072 + j + 7] * k7
        a5 += s_input[base + 3840 + j + 7] * k7
        a6 += s_input[base + 4608 + j + 7] * k7
        a7 += s_input[base + 5376 + j + 7] * k7
        j += 8
    while j < kernel_size:
        kk = s_kernel[j]
        a0 += s_input[base + 0 + j] * kk
        a1 += s_input[base + 768 + j] * kk
        a2 += s_input[base + 1536 + j] * kk
        a3 += s_input[base + 2304 + j] * kk
        a4 += s_input[base + 3072 + j] * kk
        a5 += s_input[base + 3840 + j] * kk
        a6 += s_input[base + 4608 + j] * kk
        a7 += s_input[base + 5376 + j] * kk
        j += 1

    o0 = block_start + tid
    if o0 + 0 < output_size:
        output[o0 + 0] = a0
    if o0 + 768 < output_size:
        output[o0 + 768] = a1
    if o0 + 1536 < output_size:
        output[o0 + 1536] = a2
    if o0 + 2304 < output_size:
        output[o0 + 2304] = a3
    if o0 + 3072 < output_size:
        output[o0 + 3072] = a4
    if o0 + 3840 < output_size:
        output[o0 + 3840] = a5
    if o0 + 4608 < output_size:
        output[o0 + 4608] = a6
    if o0 + 5376 < output_size:
        output[o0 + 5376] = a7


def solve(input, kernel, output, input_size, kernel_size):
    if input is None or kernel is None or output is None:
        return
    if input_size <= 0 or kernel_size <= 0:
        raise ValueError("input_size and kernel_size must be positive")
    if kernel_size > MAX_KERNEL_SIZE:
        raise ValueError(f"kernel_size must be <= {MAX_KERNEL_SIZE}")

    output_size = input_size - kernel_size + 1
    if output_size <= 0:
        raise ValueError("kernel_size must not exceed input_size")

    blocks = (output_size + BLOCK_OUT - 1) // BLOCK_OUT
    conv1d_kernel[blocks, THREADS](input, kernel, output, input_size,
                                   kernel_size)
    cuda.synchronize()


def _reference(inp, ker, out_n):
    """float64 valid convolution, computed in full or on a sample."""
    inp64 = inp.astype(np.float64)
    ker64 = ker.astype(np.float64)
    if float(inp.size) * ker.size <= 5e6:
        return np.correlate(inp64, ker64, mode="valid")
    idx = np.linspace(0, out_n - 1, num=500).astype(np.int64)
    ref = np.empty(idx.size)
    for t, i in enumerate(idx):
        ref[t] = float(np.dot(inp64[i:i + ker64.size], ker64))
    return ref, idx


def _run(name, inp, ker):
    n = inp.size
    kn = ker.size
    out_n = n - kn + 1
    d_in = cuda.to_device(inp)
    d_k = cuda.to_device(ker)
    d_out = cuda.device_array(out_n, dtype=np.float32)
    solve(d_in, d_k, d_out, n, kn)
    got = d_out.copy_to_host()

    ref = _reference(inp, ker, out_n)
    if isinstance(ref, tuple):
        ref, idx = ref
        got_cmp = got[idx]
        note = " (sampled)"
    else:
        got_cmp = got
        note = ""
    err = np.abs(got_cmp - ref)
    max_err = float(err.max()) if err.size else 0.0
    max_ref = float(np.abs(ref).max()) if ref.size else 0.0
    rel = max_err / max(1e-30, max_ref)
    ok = rel < 1e-4
    print(f"{name:>18} n={n} k={kn} out={out_n}: max_err={max_err:.3e} "
          f"rel={rel:.3e} {'OK' if ok else 'FAIL'}{note}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    # Challenge examples.
    ok &= _run("example 1", np.array([1, 2, 3, 4, 5], dtype=np.float32),
               np.array([1, 0, -1], dtype=np.float32))
    ok &= _run("example 2", np.array([2, 4, 6, 8], dtype=np.float32),
               np.array([0.5, 0.2], dtype=np.float32))

    for n, kn in [(1, 1), (5, 5), (100, 1), (100, 2), (1000, 3),
                  (1024, 16), (10000, 128), (65537, 257), (100000, 2047),
                  (260, 260)]:
        inp = rng.normal(0.0, 1.0, n).astype(np.float32)
        ker = rng.normal(0.0, 1.0, kn).astype(np.float32)
        ok &= _run("random", inp, ker)

    # Performance: input_size = 1,500,000, kernel_size = 2,047, median of 20.
    import statistics
    import time

    n, kn = 1500000, 2047
    out_n = n - kn + 1
    inp = rng.normal(0.0, 1.0, n).astype(np.float32)
    ker = rng.normal(0.0, 1.0, kn).astype(np.float32)
    d_in = cuda.to_device(inp)
    d_k = cuda.to_device(ker)
    d_out = cuda.device_array(out_n, dtype=np.float32)
    solve(d_in, d_k, d_out, n, kn)  # warm-up
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        solve(d_in, d_k, d_out, n, kn)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    gflops = 2.0 * out_n * kn / (ms * 1e6)
    got = d_out.copy_to_host()
    idx = np.linspace(0, out_n - 1, num=500).astype(np.int64)
    inp64 = inp.astype(np.float64)
    ker64 = ker.astype(np.float64)
    ref = np.array([float(np.dot(inp64[i:i + kn], ker64)) for i in idx])
    rel = float(np.abs(got[idx] - ref).max() / np.abs(ref).max())
    print(f"Perf n={n} k={kn}: {ms:.3f} ms, {gflops:.1f} GFLOP/s, "
          f"rel={rel:.3e}, {'OK' if rel < 1e-4 else 'FAIL'}")
