"""Numba CUDA port of int4_matmul.cu.

W4A16 matmul: y = x @ dequant(W)^T with packed INT4 weights (two nibbles per
byte, value nibble-8) and group-wise fp16 scales.

At the benchmark (4096^3) this is 137 GFLOP on ~40 MB, i.e. compute-bound.
In C++ the starter's one-thread-per-output form measures 6.1 s; the shipped
128x128-tile register-blocked GEMM measures 160 ms (0.86 TFLOP/s).

Numba cannot hold the 8x8 register tile, so this port uses the 32x32
shared-tile GEMM with one output per thread, dequantizing the weight tile into
shared memory exactly like the C++ loader.  Numba cannot index float16 arrays,
so x, scales and y are passed as int16 views and converted with libdevice
half2float / float2half_rn.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, int32
from numba.cuda import libdevice

TILE = 32


@cuda.jit(cache=True)
def w4a16_kernel(x, wq, scales, y, M, N, K, G, K2, NG):
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
        As[ty, tx] = (libdevice.half2float(x[row * K + a_col])
                      if (row < M and a_col < K) else float32(0.0))
        gk = t * TILE + ty
        if col < N and gk < K:
            byte = wq[col * K2 + (gk >> 1)]
            nib = (byte & 15) if (gk & 1) == 1 else (byte >> 4)
            Bs[ty, tx] = (float32(nib - 8)
                          * libdevice.half2float(scales[col * NG + gk // G]))
        else:
            Bs[ty, tx] = float32(0.0)
        cuda.syncthreads()
        for k in range(TILE):
            acc += As[ty, k] * Bs[k, tx]
        cuda.syncthreads()

    if row < M and col < N:
        y[row * N + col] = libdevice.float2half_rn(acc)


def solve(x, wq, scales, y, M, N, K, group_size):
    """x, scales, y are int16 views of float16 device arrays; wq is uint8."""
    if M <= 0 or N <= 0 or K <= 0 or group_size <= 0:
        return
    K2 = K // 2
    NG = K // group_size
    grid = ((N + TILE - 1) // TILE, (M + TILE - 1) // TILE)
    w4a16_kernel[grid, (TILE, TILE)](x, wq, scales, y, int32(M), int32(N),
                                     int32(K), int32(group_size), int32(K2),
                                     int32(NG))
    cuda.synchronize()


def _dequant_np(wq, sc, N, K, G):
    K2 = K // 2
    NG = K // G
    high = ((wq.astype(np.int32) >> 4) & 0xF) - 8
    low = (wq.astype(np.int32) & 0xF) - 8
    W = np.empty((N, K), np.float64)
    W[:, 0::2] = high * sc.astype(np.float64).repeat(G, axis=1)[:, 0::2]
    W[:, 1::2] = low * sc.astype(np.float64).repeat(G, axis=1)[:, 1::2]
    return W


def _run(name, x, wq, sc, M, N, K, G):
    x64 = x.astype(np.float64)
    W = _dequant_np(wq, sc, N, K, G)
    ref = x64 @ W.T
    dx = cuda.to_device(np.ascontiguousarray(x, dtype=np.float16)
                        .reshape(-1).view(np.int16))
    dw = cuda.to_device(np.ascontiguousarray(wq, dtype=np.uint8).reshape(-1))
    ds = cuda.to_device(np.ascontiguousarray(sc, dtype=np.float16)
                        .reshape(-1).view(np.int16))
    dy = cuda.device_array(M * N, dtype=np.int16)
    solve(dx, dw, ds, dy, M, N, K, G)
    got = dy.copy_to_host().view(np.float16).astype(np.float64).reshape(M, N)
    abs_err = np.abs(got - ref)
    bad = int(np.count_nonzero(abs_err > 0.02 + 0.01 * np.abs(ref)))
    ok = bad == 0
    print(f"{name:>16} M={M} N={N} K={K} G={G}: max_abs={np.max(abs_err):.2e} "
          f"bad={bad} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    def make(M, N, K, G):
        x = rng.uniform(-1.0, 1.0, (M, K)).astype(np.float16)
        wq = rng.integers(0, 256, (N, K // 2), dtype=np.uint8)
        sc = rng.uniform(0.01, 0.1, (N, K // G)).astype(np.float16)
        return x, wq, sc

    # Challenge example.
    ok &= _run("example",
               np.array([[1, 0, 1, 0], [0, 1, 0, 1]], np.float16),
               np.array([[0x99, 0x99], [0xAA, 0xAA], [0x77, 0x77],
                         [0x88, 0x88]], np.uint8),
               np.full((4, 2), 0.5, np.float16), 2, 4, 4, 2)
    for M, N, K in ((1, 1, 2), (3, 5, 4), (16, 16, 16), (33, 17, 32),
                    (64, 64, 64), (128, 32, 128)):
        for G in (2, 8, 32, 128):
            if K % G:
                continue
            x, wq, sc = make(M, N, K, G)
            ok &= _run(f"random {M}x{N}x{K}", x, wq, sc, M, N, K, G)

    # Benchmark: M=N=K=4096, G=128 (sampled CPU reference).
    import statistics
    import time

    M = N = K = 4096
    G = 128
    x, wq, sc = make(M, N, K, G)
    dx = cuda.to_device(x.reshape(-1).view(np.int16))
    dw = cuda.to_device(np.ascontiguousarray(wq).reshape(-1))
    ds = cuda.to_device(sc.reshape(-1).view(np.int16))
    dy = cuda.device_array(M * N, dtype=np.int16)
    solve(dx, dw, ds, dy, M, N, K, G)  # warm-up + JIT compile
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        solve(dx, dw, ds, dy, M, N, K, G)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    got = dy.copy_to_host().view(np.float16).astype(np.float64).reshape(M, N)
    W = _dequant_np(wq, sc, N, K, G)
    bad = 0
    for _ in range(50):
        m = int(rng.integers(0, M)); n = int(rng.integers(0, N))
        r = float(x[m].astype(np.float64) @ W[n])
        if abs(got[m, n] - r) > 0.02 + 0.01 * abs(r):
            bad += 1
    flops = 2.0 * M * N * K
    print(f"Perf {M}x{N}x{K} G={G}: {ms:.4f} ms, "
          f"{flops / (ms * 1e-3) / 1e12:.3f} TFLOP/s, sampled bad={bad}, "
          f"{'OK' if bad == 0 else 'FAIL'}")
    ok &= bad == 0
    print("ALL OK" if ok else "FAILURES")
