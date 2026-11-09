"""Numba CUDA port of rope_embedding.cu.

RoPE with the half-split layout:
  out[j] = q[j]*cos[j] - q[j+D/2]*sin[j]
  out[j+D/2] = q[j+D/2]*cos[j+D/2] + q[j]*sin[j+D/2]

At the benchmark (M = 1,048,576, D = 128) the starter reads all of Q, cos and
sin plus writes the output (2.15 GB) and measures 27.8 ms.  The C++ kernel
uses the guaranteed cos/sin half-split invariance to load only half of each
(1.61 GB) and measures 20.0 ms.  This port keeps the plain elementwise
mapping and reads cos/sin in full (Numba has no float4 type), so its traffic
is the starter's 2.15 GB.

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, int32, int64

THREADS = 256


@cuda.jit(cache=True)
def rope_kernel(Q, C, S, O, total, D):
    idx = int64(cuda.blockIdx.x) * THREADS + int64(cuda.threadIdx.x)
    stride = int64(cuda.blockDim.x) * int64(cuda.gridDim.x)
    half = D // 2
    for i in range(idx, total, stride):
        col = i % D
        q = Q[i]
        rot = -Q[i + half] if col < half else Q[i - half]
        O[i] = q * C[i] + rot * S[i]


def solve(Q, C, S, output, M, D):
    """Q, C, S, output are float32 device arrays of M*D elements."""
    if M <= 0 or D <= 0:
        return
    if Q is None or C is None or S is None or output is None:
        return
    total = M * D
    blocks = (total + THREADS - 1) // THREADS
    if blocks > 4096:
        blocks = 4096
    rope_kernel[blocks, THREADS](Q, C, S, output, int64(total), int32(D))
    cuda.synchronize()


def _ref(q, c, s, M, D):
    q64 = q.astype(np.float64).reshape(M, D)
    c64 = c.astype(np.float64).reshape(M, D)
    s64 = s.astype(np.float64).reshape(M, D)
    half = D // 2
    rot = np.concatenate([-q64[:, half:], q64[:, :half]], axis=1)
    return (q64 * c64 + rot * s64).reshape(-1)


def _run(name, q, c, s, M, D):
    ref = _ref(q, c, s, M, D)
    dq = cuda.to_device(np.ascontiguousarray(q, dtype=np.float32).reshape(-1))
    dc = cuda.to_device(np.ascontiguousarray(c, dtype=np.float32).reshape(-1))
    ds = cuda.to_device(np.ascontiguousarray(s, dtype=np.float32).reshape(-1))
    do = cuda.device_array(M * D, dtype=np.float32)
    solve(dq, dc, ds, do, M, D)
    got = do.copy_to_host().astype(np.float64)
    abs_err = np.abs(got - ref)
    bad = int(np.count_nonzero(abs_err > 1e-4 + 1e-4 * np.abs(ref)))
    ok = bad == 0
    print(f"{name:>16} M={M} D={D}: max_abs={np.max(abs_err):.2e} bad={bad} "
          f"{'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)

    def make(M, D):
        q = rng.uniform(-2.0, 2.0, (M, D)).astype(np.float32)
        ang = rng.uniform(0.0, 6.28, (M, D // 2)).astype(np.float32)
        c = np.concatenate([np.cos(ang), np.cos(ang)], axis=1).astype(np.float32)
        s = np.concatenate([np.sin(ang), np.sin(ang)], axis=1).astype(np.float32)
        return q.reshape(-1), c.reshape(-1), s.reshape(-1)

    ok = True
    ok &= _run("example 1", np.array([1, 2, 3, 4, 1, 1, 1, 1], np.float32),
               np.array([1, 1, 1, 1, 0, 0, 0, 0], np.float32),
               np.array([0, 0, 0, 0, 1, 1, 1, 1], np.float32), 2, 4)
    for M, D in ((1, 2), (3, 6), (1, 8), (1, 96), (5, 16), (17, 64),
                 (100, 128), (3, 130), (64, 4096)):
        q, c, s = make(M, D)
        ok &= _run(f"random {M}x{D}", q, c, s, M, D)

    # Benchmark: M = 1,048,576, D = 128.
    import statistics
    import time

    M, D = 1048576, 128
    q, c, s = make(M, D)
    dq = cuda.to_device(q)
    dc = cuda.to_device(c)
    ds = cuda.to_device(s)
    do = cuda.device_array(M * D, dtype=np.float32)
    solve(dq, dc, ds, do, M, D)  # warm-up, also triggers the JIT compile
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        solve(dq, dc, ds, do, M, D)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    ref = _ref(q, c, s, M, D)
    got = do.copy_to_host().astype(np.float64)
    bad = int(np.count_nonzero(np.abs(got - ref) > 1e-4 + 1e-4 * np.abs(ref)))
    bytes_moved = 4 * M * D * 4
    print(f"Perf {M}x{D}: {ms:.4f} ms, "
          f"{bytes_moved / (ms * 1e6):.1f} GB/s (4-array traffic), "
          f"bad={bad}, {'OK' if bad == 0 else 'FAIL'}")
    ok &= bad == 0
    print("ALL OK" if ok else "FAILURES")
