"""Numba port of ../cuda-basics/conv-1d/conv-1d.cu

Uses Numba dynamic shared memory: cuda.shared.array(0, float32) plus the
sharedmem launch argument.
"""

import numpy as np
from numba import cuda, float32


@cuda.jit
def conv1d_shared_kernel(input, kernel, output, N, kN):
    s_input = cuda.shared.array(0, float32)

    tid = cuda.threadIdx.x
    out_idx = cuda.blockIdx.x * cuda.blockDim.x + tid

    if out_idx < N:
        s_input[tid] = input[out_idx]

    if tid < kN - 1:
        halo_idx = out_idx + cuda.blockDim.x
        if halo_idx < N:
            s_input[tid + cuda.blockDim.x] = input[halo_idx]

    cuda.syncthreads()

    out_n = N - kN + 1
    if out_idx < out_n:
        total = 0.0
        for a in range(kN):
            total += s_input[tid + a] * kernel[a]
        output[out_idx] = total


def solve(input, kernel, output, N, kN):
    out_n = N - kN + 1
    threads = 256
    blocks = (out_n + threads - 1) // threads
    shared_bytes = (threads + kN - 1) * 4
    conv1d_shared_kernel[blocks, threads, 0, shared_bytes](input, kernel, output, N, kN)
    cuda.synchronize()


def test():
    rng = np.random.default_rng(0)
    N, kN = 1003, 5
    x = rng.standard_normal(N, dtype=np.float32)
    k = rng.standard_normal(kN, dtype=np.float32)
    out = np.empty(N - kN + 1, dtype=np.float32)
    dx, dk = cuda.to_device(x), cuda.to_device(k)
    dout = cuda.to_device(out)
    solve(dx, dk, dout, N, kN)
    ref = np.correlate(x.astype(np.float64), k.astype(np.float64), mode="valid")
    np.testing.assert_allclose(dout.copy_to_host(), ref, rtol=1e-5, atol=1e-5)
    print("PASS conv-1d")


if __name__ == "__main__":
    test()
