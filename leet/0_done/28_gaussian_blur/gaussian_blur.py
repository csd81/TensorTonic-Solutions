"""Numba CUDA port of gaussian_blur.cu.

output[i,j] = sum input[i+m-ph, j+n-pw] * kernel[m,n] with zero padding.
Benchmark: 512x512 image, 7x7 kernel.  The C++ kernel measures ~58 us
against a 32 us pure-copy floor (the starter is 212 us); this port measures
234 us -- slightly slower than the C++ *naive* kernel, because Numba's
shared-memory codegen cannot specialize on the kernel width at compile time
and keeps a runtime inner loop.

Structure mirrors the C++ kernel: dynamic shared tile sized from the actual
halo, 16x16 threads, and 4 adjacent output columns per thread.  Two
Numba-specific points:

  * Numba cannot specialize a kernel on the kernel width at runtime the way
    the C++ switch over template instantiations does, so the kc loop runs at
    runtime.  To keep shared-memory traffic at one load per 4 multiply-adds
    anyway, the window is kept in a *rolling* set of four registers
    (w0..w3): each kc step issues one shared load (the next window value)
    and shifts the registers, exactly like a classic 1-D stencil.
  * The shared array is dynamic: `cuda.shared.array(0, dtype)` with the byte
    size passed at launch as the fourth launch-configuration item.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32, int32

TX = 16
TY = 16
CPT = 4
BW = TX * CPT  # 64 output columns per block
BH = 16


@cuda.jit(cache=True)
def blur_kernel(input, kernel, output, R, C, KR, KC):
    tile = cuda.shared.array(0, dtype=float32)

    # Module-level Python ints are int64 in Numba, and mixing them into index
    # arithmetic promotes whole expressions to 64-bit (slow on sm_61).  Cast
    # the derived values once so the hot index math stays int32.
    ph = int32(KR >> 1)
    pw = int32(KC >> 1)
    pitch = int32(BW) + int32(2) * pw
    rows = int32(BH) + int32(2) * ph
    r0 = int32(cuda.blockIdx.y) * int32(BH) - ph
    c0 = int32(cuda.blockIdx.x) * int32(BW) - pw

    # Nested (row, col) loops: a flat index would need idx // pitch and
    # idx % pitch with a runtime divisor, which Numba lowers to a 64-bit
    # division in every iteration.
    i = int32(cuda.threadIdx.y)
    while i < rows:
        gr = r0 + i
        j = int32(cuda.threadIdx.x)
        while j < pitch:
            gc = c0 + j
            if gr >= 0 and gr < R and gc >= 0 and gc < C:
                tile[i * pitch + j] = input[gr * C + gc]
            else:
                tile[i * pitch + j] = float32(0.0)
            j += int32(TX)
        i += int32(TY)
    cuda.syncthreads()

    orow = int32(cuda.blockIdx.y) * int32(BH) + int32(cuda.threadIdx.y)
    ocol = (int32(cuda.blockIdx.x) * int32(BW) +
            int32(cuda.threadIdx.x) * int32(CPT))
    if orow < R:
        a0 = float32(0.0)
        a1 = float32(0.0)
        a2 = float32(0.0)
        a3 = float32(0.0)
        tr = int32(cuda.threadIdx.y)
        tc = int32(cuda.threadIdx.x) * int32(CPT)
        kr = int32(0)
        while kr < KR:
            base = (tr + kr) * pitch + tc
            kb = kr * KC
            w0 = tile[base]
            w1 = tile[base + 1]
            w2 = tile[base + 2]
            w3 = tile[base + 3]
            kc = int32(0)
            while kc < KC:
                kv = kernel[kb + kc]
                a0 += w0 * kv
                a1 += w1 * kv
                a2 += w2 * kv
                a3 += w3 * kv
                w0 = w1
                w1 = w2
                w2 = w3
                w3 = tile[base + kc + 4]
                kc += int32(1)
            kr += int32(1)

        o = orow * C + ocol
        if ocol < C:
            output[o] = a0
        if ocol + 1 < C:
            output[o + 1] = a1
        if ocol + 2 < C:
            output[o + 2] = a2
        if ocol + 3 < C:
            output[o + 3] = a3


def solve(input, kernel, output, R, C, KR, KC):
    """input (R*C), kernel (KR*KC), output (R*C): float32 device arrays."""
    if R <= 0 or C <= 0 or KR <= 0 or KC <= 0:
        return
    grid = ((C + BW - 1) // BW, (R + BH - 1) // BH)
    smem = (BH + 2 * (KR // 2)) * (BW + 2 * (KC // 2)) * 4
    blur_kernel[grid, (TX, TY), 0, smem](input, kernel, output, int32(R),
                                         int32(C), int32(KR), int32(KC))
    cuda.synchronize()


def _ref(a, k, R, C, KR, KC):
    x = np.zeros((R + KR - 1, C + KC - 1), dtype=np.float64)
    ph, pw = KR // 2, KC // 2
    x[ph:ph + R, pw:pw + C] = a.astype(np.float64).reshape(R, C)
    out = np.zeros((R, C), dtype=np.float64)
    for kr in range(KR):
        for kc in range(KC):
            out += k[kr * KC + kc] * x[kr:kr + R, kc:kc + C]
    return out


def _check(name, a, k, R, C, KR, KC):
    ref = _ref(a, k, R, C, KR, KC)
    din = cuda.to_device(np.ascontiguousarray(a, dtype=np.float32).reshape(-1))
    dk = cuda.to_device(np.ascontiguousarray(k, dtype=np.float32).reshape(-1))
    dout = cuda.device_array(R * C, dtype=np.float32)
    solve(din, dk, dout, R, C, KR, KC)
    got = dout.copy_to_host().astype(np.float64).reshape(R, C)
    err = np.abs(got - ref)
    bad = int(np.count_nonzero(~(err <= 1e-4 + 1e-4 * np.abs(ref))))
    print(f"{name:>22} img={R}x{C} k={KR}x{KC}: bad={bad} "
          f"max_abs={err.max():.2e} {'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)

    def make(R, C, KR, KC):
        a = rng.uniform(-1, 1, R * C).astype(np.float32)
        k = np.abs(rng.uniform(-1, 1, KR * KC)).astype(np.float32)
        k /= k.sum()
        return a, k

    ok = True
    ok &= _check("example 1",
                 np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20, 21, 22, 23, 24, 25], np.float32),
                 np.array([0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625,
                           0.125, 0.0625], np.float32), 5, 5, 3, 3)
    for R, C, KR, KC in ((1, 1, 3, 3), (1, 7, 7, 3), (100, 37, 5, 3),
                         (64, 64, 21, 21), (33, 65, 9, 13), (512, 512, 21, 7),
                         (2048, 2048, 5, 5), (4096, 4096, 3, 3)):
        a, k = make(R, C, KR, KC)
        ok &= _check(f"{R}x{C} k{KR}x{KC}", a, k, R, C, KR, KC)

    # Benchmark: 512 x 512, 7 x 7.
    import statistics
    import time

    R, C, KR, KC = 512, 512, 7, 7
    a, k = make(R, C, KR, KC)
    ok &= _check("benchmark shape", a, k, R, C, KR, KC)
    din = cuda.to_device(a)
    dk = cuda.to_device(k)
    dout = cuda.device_array(R * C, dtype=np.float32)
    solve(din, dk, dout, R, C, KR, KC)  # warm-up / JIT
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        solve(din, dk, dout, R, C, KR, KC)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    print(f"Perf img={R}x{C} k={KR}x{KC}: {ms:.4f} ms, "
          f"{2 * R * C * 4 / (ms * 1e6):.1f} GB/s")
    print("ALL OK" if ok else "FAILURES")
