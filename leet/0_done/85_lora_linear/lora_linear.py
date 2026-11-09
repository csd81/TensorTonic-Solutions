"""Numba CUDA port of lora_linear.cu.

LoRA linear: output = x·Wᵀ + lora_scale·(x·Aᵀ)·Bᵀ, with x [M,K], W [N,K],
A [R,K], B [N,R] and output [M,N].

At the benchmark (M=256, K=4096, N=4096, R=64) the base product is 8.6 of the
8.9 GFLOP.  The C++ version runs all three products through one 128x128-tile,
8x8-per-thread C = A·Bᵀ kernel and measures 9.3 ms (0.955 TFLOP/s); the
starter's untiled one-thread-per-output form measures 282 ms.

Numba cannot hold the 8x8 register tile (64 accumulators would spill to local
memory), so this port uses the 32x32 shared-memory tile with one output per
thread -- the same structure as the challenge-22 GEMM port -- for all three
products.  It is dispatch-light (three launches) but compute-bound at
Numba's scalar codegen rate.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, int32

TILE = 32
_temp = None


@cuda.jit(cache=True)
def gemm_t_kernel(A, B, C, M, N, K, scale, accum):
    """C = scale * A @ Bᵀ, or C += scale * A @ Bᵀ when accum == 1.

    A is [M, K] row-major, B is [N, K] row-major (so Bᵀ is [K, N])."""
    As = cuda.shared.array((TILE, TILE), dtype=float32)
    Bs = cuda.shared.array((TILE, TILE), dtype=float32)
    row = int32(cuda.blockIdx.y) * TILE + int32(cuda.threadIdx.y)
    col = int32(cuda.blockIdx.x) * TILE + int32(cuda.threadIdx.x)
    ty = cuda.threadIdx.y
    tx = cuda.threadIdx.x
    acc = float32(0.0)

    num_tiles = (K + TILE - 1) // TILE
    for t in range(num_tiles):
        a_col = t * TILE + tx
        As[ty, tx] = A[row * K + a_col] if (row < M and a_col < K) else float32(0.0)
        b_row = t * TILE + ty  # contraction index k for B
        Bs[ty, tx] = B[col * K + b_row] if (col < N and b_row < K) else float32(0.0)
        cuda.syncthreads()
        for k in range(TILE):
            acc += As[ty, k] * Bs[k, tx]
        cuda.syncthreads()

    if row < M and col < N:
        if accum == 1:
            C[row * N + col] += scale * acc
        else:
            C[row * N + col] = scale * acc


def _get_temp(n):
    global _temp
    if _temp is None or _temp.size < n:
        _temp = cuda.device_array(n, dtype=np.float32)
    return _temp


def solve(x, W, A, B, output, batch, d_in, d_out, rank, lora_scale):
    """x: [batch, d_in]; W: [d_out, d_in]; A: [rank, d_in]; B: [d_out, rank];
    output: [batch, d_out].  All flat float32 device arrays."""
    if batch <= 0 or d_in <= 0 or d_out <= 0 or rank <= 0:
        return
    if x is None or W is None or A is None or B is None or output is None:
        return
    temp = _get_temp(batch * rank)
    block = (TILE, TILE)
    grid_out = ((d_out + TILE - 1) // TILE, (batch + TILE - 1) // TILE)
    grid_rank = ((rank + TILE - 1) // TILE, (batch + TILE - 1) // TILE)

    gemm_t_kernel[grid_out, block](x, W, output, int32(batch), int32(d_out),
                                   int32(d_in), float32(1.0), int32(0))
    gemm_t_kernel[grid_rank, block](x, A, temp, int32(batch), int32(rank),
                                    int32(d_in), float32(1.0), int32(0))
    gemm_t_kernel[grid_out, block](temp, B, output, int32(batch), int32(d_out),
                                   int32(rank), float32(lora_scale), int32(1))
    cuda.synchronize()


def _run(name, x, W, A, B, M, K, N, R, scale):
    x64 = x.astype(np.float64).reshape(M, K)
    W64 = W.astype(np.float64).reshape(N, K)
    A64 = A.astype(np.float64).reshape(R, K)
    B64 = B.astype(np.float64).reshape(N, R)
    ref = x64 @ W64.T + scale * (x64 @ A64.T) @ B64.T

    dx = cuda.to_device(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))
    dW = cuda.to_device(np.ascontiguousarray(W, dtype=np.float32).reshape(-1))
    dA = cuda.to_device(np.ascontiguousarray(A, dtype=np.float32).reshape(-1))
    dB = cuda.to_device(np.ascontiguousarray(B, dtype=np.float32).reshape(-1))
    dout = cuda.device_array(M * N, dtype=np.float32)
    solve(dx, dW, dA, dB, dout, M, K, N, R, scale)
    got = dout.copy_to_host().reshape(M, N).astype(np.float64)
    abs_err = np.abs(got - ref)
    bad = int(np.count_nonzero(abs_err > 1e-3 + 1e-4 * np.abs(ref)))
    ok = bad == 0
    print(f"{name:>18} M={M} K={K} N={N} R={R}: max_abs={np.max(abs_err):.2e} "
          f"bad={bad} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    def rnd(m, n):
        return rng.uniform(-1.0, 1.0, (m, n)).astype(np.float32).reshape(-1)

    ok &= _run("example",
               np.array([1, 0, -1, 2, 0, 1, 1, -1], np.float32),
               np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], np.float32),
               np.array([1, 0, 0, 0, 0, 1, 0, 0], np.float32),
               np.array([1, 0, 0, 1, 0, 0], np.float32), 2, 4, 3, 2, 0.5)
    for M, K, N, R in ((1, 1, 1, 1), (3, 10, 7, 2), (17, 33, 19, 5),
                       (64, 128, 64, 16), (100, 257, 129, 64)):
        ok &= _run(f"random {M}x{K}x{N} r{R}", rnd(M, K), rnd(N, K), rnd(R, K),
                   rnd(N, R), M, K, N, R, 0.5)

    # Benchmark: batch=256, d_in=d_out=4096, rank=64.
    import statistics
    import time

    M, K, N, R = 256, 4096, 4096, 64
    x, W, A, B = rnd(M, K), rnd(N, K), rnd(R, K), rnd(N, R)
    dx = cuda.to_device(x)
    dW = cuda.to_device(W)
    dA = cuda.to_device(A)
    dB = cuda.to_device(B)
    dout = cuda.device_array(M * N, dtype=np.float32)
    solve(dx, dW, dA, dB, dout, M, K, N, R, 0.5)  # warm-up + JIT compile
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        solve(dx, dW, dA, dB, dout, M, K, N, R, 0.5)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    ref = (x.astype(np.float64).reshape(M, K) @ W.astype(np.float64).reshape(N, K).T
           + 0.5 * ((x.astype(np.float64).reshape(M, K)
                     @ A.astype(np.float64).reshape(R, K).T)
                    @ B.astype(np.float64).reshape(N, R).T))
    got = dout.copy_to_host().reshape(M, N).astype(np.float64)
    bad = int(np.count_nonzero(np.abs(got - ref) > 1e-3 + 1e-4 * np.abs(ref)))
    flops = 2.0 * (M * K * N + M * K * R + M * N * R)
    print(f"Perf {M}x{K}x{N} r{R}: {ms:.4f} ms, "
          f"{flops / (ms * 1e-3) / 1e12:.3f} TFLOP/s, bad={bad}, "
          f"{'OK' if bad == 0 else 'FAIL'}")
    ok &= bad == 0
    print("ALL OK" if ok else "FAILURES")
