"""Numba CUDA port of causal_depthwise_conv1d.cu.

Causal depthwise 1D convolution over a channels-last (B, L, D) tensor:

    out[b, l, d] = bias[d] + sum_{k<K} weight[d, k] * x[b, l-k, d]

with x[b, l-k, d] = 0 for l-k < 0.

The C++ kernel gives each thread four channels (float4) and a tile of
sequence positions, keeping a circular window in registers so every input is
loaded exactly once.  Numba has neither vector loads nor register arrays, so
this port uses one thread per channel and a tile of TILE_L consecutive
sequence positions, with K scalar loads per output.  Consecutive threads
handle consecutive d, so every load and store is coalesced, and because a
thread walks l in order, the K rows it reads for step s+1 are the same rows
it read for step s -- that reuse is served by the block's L1 rather than by
L2, so DRAM traffic stays at read-once + write-once.

solve(x, weight, bias, output, B, L, D, K) keeps the challenge signature.
Arrays may be Numba device arrays (preferred) or host NumPy arrays.
"""

import numpy as np
from numba import cuda, float32

THREADS = 256
TILE_L = 16


@cuda.jit
def conv_kernel(x, weight, bias, output, L, D, K):
    d = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if d < D:
        b = cuda.blockIdx.z
        l0 = cuda.blockIdx.y * TILE_L
        base = (b * L + l0) * D + d
        bv = bias[d]  # hoisted out of the sequence loop
        for s in range(TILE_L):
            l = l0 + s
            if l >= L:
                break
            acc = bv
            for k in range(K):
                if l >= k:
                    acc += weight[d * K + k] * x[base + s * D - k * D]
            output[base + s * D] = acc


def solve(x, weight, bias, output, B, L, D, K):
    if x is None or weight is None or bias is None or output is None:
        return
    if B <= 0 or L <= 0 or D <= 0 or K <= 0:
        raise ValueError("dimensions must be positive")

    X = x if getattr(x, "ndim", 1) == 1 else x.reshape(B * L * D)
    W = weight if getattr(weight, "ndim", 1) == 1 else weight.reshape(D * K)
    Bs = bias if getattr(bias, "ndim", 1) == 1 else bias.reshape(D)
    O = output if getattr(output, "ndim", 1) == 1 else output.reshape(B * L * D)

    grid = ((D + THREADS - 1) // THREADS, (L + TILE_L - 1) // TILE_L, B)
    conv_kernel[grid, THREADS](X, W, Bs, O, L, D, K)
    cuda.synchronize()


def _reference(x, w, bias, B, L, D, K):
    x = x.astype(np.float64).reshape(B, L, D)
    w = w.astype(np.float64).reshape(D, K)
    bias = bias.astype(np.float64).reshape(D)
    out = np.zeros((B, L, D), dtype=np.float64)
    out += bias[None, None, :]
    for k in range(K):
        if k == 0:
            out += w[:, k][None, None, :] * x
        else:
            out[:, k:, :] += (w[:, k][None, None, :] * x[:, :L - k, :])
    return out.reshape(-1)


def _run(name, x, w, bias, B, L, D, K):
    ref = _reference(x, w, bias, B, L, D, K)
    dX = cuda.to_device(np.ascontiguousarray(x, dtype=np.float32))
    dW = cuda.to_device(np.ascontiguousarray(w, dtype=np.float32))
    dB = cuda.to_device(np.ascontiguousarray(bias, dtype=np.float32))
    dO = cuda.device_array(B * L * D, dtype=np.float32)
    solve(dX, dW, dB, dO, B, L, D, K)
    got = dO.copy_to_host().astype(np.float64)
    err = np.abs(got - ref)
    max_err = float(err.max()) if err.size else 0.0
    max_ref = float(np.abs(ref).max()) if ref.size else 0.0
    rel = max_err / max(1e-30, max_ref)
    ok = rel < 1e-6
    print(f"{name:>22} B={B} L={L} D={D} K={K}: max_err={max_err:.3e} "
          f"rel={rel:.3e} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    # Challenge example.
    ok &= _run("example",
               np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32),
               np.array([1, 0, -1, 1, 1, 1], dtype=np.float32),
               np.array([0, 0], dtype=np.float32), 1, 4, 2, 3)

    def random_case(B, L, D, K, name):
        x = rng.uniform(-1, 1, B * L * D).astype(np.float32)
        w = rng.uniform(-1, 1, D * K).astype(np.float32)
        bias = rng.uniform(-1, 1, D).astype(np.float32)
        return _run(name, x, w, bias, B, L, D, K)

    ok &= random_case(1, 1, 4, 1, "single element")
    ok &= random_case(1, 1, 4, 3, "L=1 K=3")
    ok &= random_case(1, 5, 3, 2, "D not divisible by 4")
    ok &= random_case(1, 7, 4, 8, "K=8")
    ok &= random_case(2, 33, 8, 4, "small")
    ok &= random_case(3, 129, 12, 5, "D=12")
    ok &= random_case(1, 1000, 17, 3, "D=17 odd")
    ok &= random_case(16, 64, 256, 4, "B=16")
    ok &= random_case(2, 300, 1000, 3, "D=1000")
    ok &= random_case(3, 2048, 4096, 4, "benchmark shape")

    # Performance: B=8, L=2048, D=4096, K=4 and K=3.
    import statistics
    import time

    B, L, D = 8, 2048, 4096
    for K in (4, 3):
        x = rng.uniform(-1, 1, B * L * D).astype(np.float32)
        w = rng.uniform(-1, 1, D * K).astype(np.float32)
        bias = rng.uniform(-1, 1, D).astype(np.float32)
        dX = cuda.to_device(x)
        dW = cuda.to_device(w)
        dB = cuda.to_device(bias)
        dO = cuda.device_array(B * L * D, dtype=np.float32)
        solve(dX, dW, dB, dO, B, L, D, K)  # warm-up
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            solve(dX, dW, dB, dO, B, L, D, K)
            times.append((time.perf_counter() - t0) * 1000.0)
        ms = statistics.median(times)
        out = dO.copy_to_host()
        ref = _reference(x, w, bias, B, L, D, K)
        rel = float(np.abs(out.astype(np.float64) - ref).max() /
                    max(1e-30, np.abs(ref).max()))
        gbps = 2.0 * B * L * D * 4 / (ms * 1e6)
        print(f"Perf B={B} L={L} D={D} K={K}: {ms:.3f} ms, "
              f"{gbps:.1f} GB/s (2x N bytes), rel={rel:.2e}, "
              f"{'OK' if rel < 1e-6 else 'FAIL'}")
        ok &= rel < 1e-6
    print("ALL OK" if ok else "FAILURES")
