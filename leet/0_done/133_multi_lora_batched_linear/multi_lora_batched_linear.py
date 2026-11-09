"""Numba CUDA port of multi_lora_batched_linear.cu.

y_t = x_t W + scale * (x_t A_{a_t}) B_{a_t} for adapter_ids[t] = a_t >= 0;
base-only for a_t = -1.  x [T,K], W [K,N], A [L,K,R] (A_a[k,r] row-major),
B [L,R,N] (B_a[r,n] row-major), y [T,N].  Benchmark: T=1024, K=4096, N=4096,
R=16, L=8 (34.4 GFLOP, dominated by the base product).

Structure mirrors the C++ kernel:

  * Base product, register-blocked GEMM.  Numba cannot hold an 8x8 accumulator
    tile as a value array without spilling to local memory, so this port uses
    a 64x64 block tile with 16 explicit scalar accumulators per thread
    (4 rows x 4 interleaved columns, 16x16 threads).  Shared reads per FMA
    drop from 2 (one accumulator) to 0.5; the C++ 128x128 tile with 8x8
    accumulators keeps 0.25.
  * LoRA path, one block per token: the R-vector temp = x_t A_a is computed
    cooperatively into shared memory (lanes split over both r and chunks of
    K, partials reduced in shared), then applied to the token's N columns.
    The K-loop runs once per token, not once per output element.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, int32

# GEMM tile shape.
BM = 64
BN = 64
BK = 32
BKP = BK + 1  # shared-array strides must be literal global constants
BNP = BN + 1
TX = 16
TY = 16
TPB = TX * TY

# LoRA kernel shape.
LORA_THREADS = 128
MAX_R = LORA_THREADS  # challenge caps R at 64


@cuda.jit(cache=True)
def base_gemm_kernel(x, W, y, T, K, N):
    As = cuda.shared.array((BM, BKP), dtype=float32)
    Bs = cuda.shared.array((BK, BNP), dtype=float32)

    m0 = cuda.blockIdx.y * BM
    n0 = cuda.blockIdx.x * BN
    tid = cuda.threadIdx.y * TX + cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tx = cuda.threadIdx.x

    a00 = float32(0.0)
    a01 = float32(0.0)
    a02 = float32(0.0)
    a03 = float32(0.0)
    a10 = float32(0.0)
    a11 = float32(0.0)
    a12 = float32(0.0)
    a13 = float32(0.0)
    a20 = float32(0.0)
    a21 = float32(0.0)
    a22 = float32(0.0)
    a23 = float32(0.0)
    a30 = float32(0.0)
    a31 = float32(0.0)
    a32 = float32(0.0)
    a33 = float32(0.0)

    t = int32(0)
    while t < K:
        idx = tid
        while idx < BM * BK:
            row = idx // BK
            k = idx % BK
            gm = m0 + row
            gk = t + k
            if gm < T and gk < K:
                As[row, k] = x[gm * K + gk]
            else:
                As[row, k] = float32(0.0)
            idx += TPB
        idx = tid
        while idx < BK * BN:
            kb = idx // BN
            col = idx % BN
            gk = t + kb
            gn = n0 + col
            if gk < K and gn < N:
                Bs[kb, col] = W[gk * N + gn]
            else:
                Bs[kb, col] = float32(0.0)
            idx += TPB
        cuda.syncthreads()

        # 4 rows x 4 interleaved columns per thread.
        for k in range(int32(BK)):
            r0 = As[ty, k]
            r1 = As[ty + 16, k]
            r2 = As[ty + 32, k]
            r3 = As[ty + 48, k]
            c0 = Bs[k, tx]
            c1 = Bs[k, tx + 16]
            c2 = Bs[k, tx + 32]
            c3 = Bs[k, tx + 48]
            a00 += r0 * c0
            a01 += r0 * c1
            a02 += r0 * c2
            a03 += r0 * c3
            a10 += r1 * c0
            a11 += r1 * c1
            a12 += r1 * c2
            a13 += r1 * c3
            a20 += r2 * c0
            a21 += r2 * c1
            a22 += r2 * c2
            a23 += r2 * c3
            a30 += r3 * c0
            a31 += r3 * c1
            a32 += r3 * c2
            a33 += r3 * c3
        cuda.syncthreads()

        t += int32(BK)

    gm0 = m0 + ty
    gm1 = m0 + ty + 16
    gm2 = m0 + ty + 32
    gm3 = m0 + ty + 48
    gn0 = n0 + tx
    gn1 = n0 + tx + 16
    gn2 = n0 + tx + 32
    gn3 = n0 + tx + 48
    if gm0 < T:
        if gn0 < N:
            y[gm0 * N + gn0] = a00
        if gn1 < N:
            y[gm0 * N + gn1] = a01
        if gn2 < N:
            y[gm0 * N + gn2] = a02
        if gn3 < N:
            y[gm0 * N + gn3] = a03
    if gm1 < T:
        if gn0 < N:
            y[gm1 * N + gn0] = a10
        if gn1 < N:
            y[gm1 * N + gn1] = a11
        if gn2 < N:
            y[gm1 * N + gn2] = a12
        if gn3 < N:
            y[gm1 * N + gn3] = a13
    if gm2 < T:
        if gn0 < N:
            y[gm2 * N + gn0] = a20
        if gn1 < N:
            y[gm2 * N + gn1] = a21
        if gn2 < N:
            y[gm2 * N + gn2] = a22
        if gn3 < N:
            y[gm2 * N + gn3] = a23
    if gm3 < T:
        if gn0 < N:
            y[gm3 * N + gn0] = a30
        if gn1 < N:
            y[gm3 * N + gn1] = a31
        if gn2 < N:
            y[gm3 * N + gn2] = a32
        if gn3 < N:
            y[gm3 * N + gn3] = a33


@cuda.jit(cache=True)
def lora_kernel(x, A, B, adapter_ids, y, T, K, N, R, scale):
    temp = cuda.shared.array(MAX_R, dtype=float32)
    partial = cuda.shared.array(LORA_THREADS, dtype=float32)

    t = int32(cuda.blockIdx.x)
    a = adapter_ids[t]
    if a >= 0:
        base_x = t * K
        base_A = a * K * R
        base_B = a * R * N
        tid = int32(cuda.threadIdx.x)

        # temp[r] = sum_k x_t[k] * A_a[k, r]; lanes split over chunks of K.
        lanes = int32(LORA_THREADS) // R
        r = tid % R
        chunk = tid // R
        s = float32(0.0)
        if chunk < lanes:
            k = chunk
            while k < K:
                s += x[base_x + k] * A[base_A + k * R + r]
                k += lanes
        partial[tid] = s
        cuda.syncthreads()
        if tid < R:
            tot = float32(0.0)
            c = int32(0)
            while c < lanes:
                tot += partial[c * R + tid]
                c += int32(1)
            temp[tid] = tot
        cuda.syncthreads()

        # y[t, n] += scale * sum_r temp[r] * B_a[r, n]
        n = tid
        while n < N:
            acc = float32(0.0)
            rr = int32(0)
            while rr < R:
                acc += temp[rr] * B[base_B + rr * N + n]
                rr += int32(1)
            y[t * N + n] += scale * acc
            n += int32(LORA_THREADS)


def solve(x, W, A, B, adapter_ids, y, T, K, N, R, L, scale):
    """All tensors are device arrays (float32 except int32 adapter_ids)."""
    if T <= 0 or K <= 0 or N <= 0 or R <= 0 or R > MAX_R:
        return
    grid = ((N + BN - 1) // BN, (T + BM - 1) // BM)
    base_gemm_kernel[grid, (TX, TY)](x, W, y, int32(T), int32(K), int32(N))
    lora_kernel[T, LORA_THREADS](x, A, B, adapter_ids, y, int32(T), int32(K),
                                 int32(N), int32(R), float32(scale))
    cuda.synchronize()


def _ref(x, W, A, B, ids, T, K, N, R, scale):
    y = x.astype(np.float64) @ W.astype(np.float64)
    for a in range(A.shape[0]):
        rows = np.nonzero(ids == a)[0]
        if rows.size == 0:
            continue
        temp = x[rows].astype(np.float64) @ A[a].astype(np.float64)  # [m, R]
        y[rows] += scale * (temp @ B[a].astype(np.float64))
    return y


def _check(name, x, W, A, B, ids, T, K, N, R, L, scale):
    ref = _ref(x, W, A, B, ids, T, K, N, R, scale)
    dx = cuda.to_device(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))
    dW = cuda.to_device(np.ascontiguousarray(W, dtype=np.float32).reshape(-1))
    da = cuda.to_device(np.ascontiguousarray(A, dtype=np.float32).reshape(-1))
    db = cuda.to_device(np.ascontiguousarray(B, dtype=np.float32).reshape(-1))
    dids = cuda.to_device(np.ascontiguousarray(ids, dtype=np.int32))
    dy = cuda.device_array(T * N, dtype=np.float32)
    solve(dx, dW, da, db, dids, dy, T, K, N, R, L, scale)
    got = dy.copy_to_host().reshape(T, N).astype(np.float64)
    err = np.abs(got - ref)
    bad = int(np.count_nonzero(~(err <= 1e-2 + 1e-3 * np.abs(ref))))
    print(f"{name:>20} T={T} K={K} N={N} R={R} L={L}: bad={bad} "
          f"max_abs={err.max():.2e} {'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)

    def make(T, K, N, R, L, ids=None, all_minus1=False):
        x = rng.uniform(-1, 1, (T, K)).astype(np.float32)
        W = rng.uniform(-1, 1, (K, N)).astype(np.float32)
        A = rng.uniform(-1, 1, (L, K, R)).astype(np.float32)
        B = rng.uniform(-1, 1, (L, R, N)).astype(np.float32)
        if ids is None:
            ids = rng.integers(-1, L, T).astype(np.int32)
        if all_minus1:
            ids = np.full(T, -1, dtype=np.int32)
        return x, W, A, B, ids

    ok = True
    # Challenge example: y = [[1.5, 2.5], [5, 2], [1, 1]].
    ok &= _check("example",
                 np.array([[1, 2], [3, 4], [1, 1]], np.float32),
                 np.array([[1, 0], [0, 1]], np.float32),
                 np.array([[[1], [0]], [[0], [1]]], np.float32),
                 np.array([[[1, 1]], [[1, -1]]], np.float32),
                 np.array([0, 1, -1], np.int32), 3, 2, 2, 1, 2, 0.5)
    for T, K, N, R, L in ((1, 1, 1, 1, 1), (5, 3, 7, 2, 3),
                          (17, 33, 19, 5, 4), (64, 128, 64, 16, 8),
                          (100, 257, 129, 64, 2), (129, 130, 131, 3, 64)):
        x, W, A, B, ids = make(T, K, N, R, L)
        ok &= _check(f"{T}x{K}x{N} r{R} l{L}", x, W, A, B, ids, T, K, N, R, L,
                     0.5)
    x, W, A, B, ids = make(64, 128, 64, 16, 8, all_minus1=True)
    ok &= _check("all ids -1", x, W, A, B, ids, 64, 128, 64, 16, 8, 0.5)

    # Benchmark: T=1024, K=4096, N=4096, R=16, L=8.
    import statistics
    import time

    T, K, N, R, L = 1024, 4096, 4096, 16, 8
    x, W, A, B, ids = make(T, K, N, R, L)
    ok &= _check("benchmark shape", x, W, A, B, ids, T, K, N, R, L, 0.5)

    dx = cuda.to_device(x.reshape(-1))
    dW = cuda.to_device(W.reshape(-1))
    da = cuda.to_device(A.reshape(-1))
    db = cuda.to_device(B.reshape(-1))
    dids = cuda.to_device(ids)
    dy = cuda.device_array(T * N, dtype=np.float32)
    solve(dx, dW, da, db, dids, dy, T, K, N, R, L, 0.5)  # warm-up / JIT
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(dx, dW, da, db, dids, dy, T, K, N, R, L, 0.5)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    flops = 2.0 * (T * K * N + T * K * R + T * R * N)
    print(f"Perf T={T} K={K} N={N} R={R} L={L}: {ms:.4f} ms, "
          f"{flops / (ms * 1e-3) / 1e12:.3f} TFLOP/s")
    print("ALL OK" if ok else "FAILURES")
