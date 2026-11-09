"""Numba CUDA port of rgb_to_grayscale.cu.

RGB -> grayscale: out[p] = 0.299*R + 0.587*G + 0.114*B, with RGB triples
stored consecutively in the input.

At the benchmark (2048 x 2048 = 4,194,304 pixels) the kernel moves 67 MB, so
it is memory-bound: in C++ the scalar kernel, the naive form and a vectorized
4-pixels-per-thread form all measure 0.80 ms (84 GB/s).  This port keeps the
scalar mapping (Numba has no float4 type).

Kernels use cache=True so the JIT cost is paid once per source change.
"""

import numpy as np
from numba import cuda, float32

THREADS = 256
CR = float32(0.299)
CG = float32(0.587)
CB = float32(0.114)


@cuda.jit(cache=True)
def rgb_to_gray_kernel(input, output, pixels):
    p = cuda.grid(1)
    if p < pixels:
        o = 3 * p
        output[p] = CR * input[o] + CG * input[o + 1] + CB * input[o + 2]


def solve(input, output, width, height):
    """input is a float32 device array of length 3*width*height; output of
    length width*height."""
    if width <= 0 or height <= 0:
        return
    if input is None or output is None:
        return
    pixels = width * height
    blocks = (pixels + THREADS - 1) // THREADS
    rgb_to_gray_kernel[blocks, THREADS](input, output, pixels)
    cuda.synchronize()


def _run(name, rgb, width, height):
    pixels = width * height
    ref = (0.299 * rgb[0::3].astype(np.float64)
           + 0.587 * rgb[1::3].astype(np.float64)
           + 0.114 * rgb[2::3].astype(np.float64))
    da = cuda.to_device(np.ascontiguousarray(rgb, dtype=np.float32))
    dout = cuda.device_array(pixels, dtype=np.float32)
    solve(da, dout, width, height)
    got = dout.copy_to_host().astype(np.float64)
    abs_err = np.abs(got - ref)
    rel = np.max(abs_err) / max(1.0, np.max(np.abs(ref)))
    ok = np.max(abs_err) < 1e-3 and rel < 1e-5
    print(f"{name:>20} {width}x{height}: max_abs={np.max(abs_err):.2e} "
          f"rel={rel:.2e} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True
    ok &= _run("example 1",
               np.array([255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0,
                         128.0, 128.0, 128.0], np.float32), 2, 2)
    ok &= _run("example 2", np.array([100.0, 150.0, 200.0], np.float32), 1, 1)
    for w, h in ((1, 5), (5, 1), (3, 7), (64, 64), (100, 100)):
        ok &= _run(f"random {w}x{h}",
                   rng.uniform(0.0, 255.0, 3 * w * h).astype(np.float32), w, h)

    # Benchmark: 2048 x 2048.
    import statistics
    import time

    W, H = 2048, 2048
    pixels = W * H
    rgb = rng.uniform(0.0, 255.0, 3 * pixels).astype(np.float32)
    da = cuda.to_device(rgb)
    dout = cuda.device_array(pixels, dtype=np.float32)
    solve(da, dout, W, H)  # warm-up, also triggers the JIT compile
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        solve(da, dout, W, H)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    ref = (0.299 * rgb[0::3].astype(np.float64)
           + 0.587 * rgb[1::3].astype(np.float64)
           + 0.114 * rgb[2::3].astype(np.float64))
    err = float(np.max(np.abs(dout.copy_to_host().astype(np.float64) - ref)))
    bytes_moved = 4 * pixels * 4
    print(f"Perf {W}x{H}: {ms:.4f} ms, {bytes_moved / (ms * 1e6):.1f} GB/s, "
          f"max_abs={err:.2e}, {'OK' if err < 1e-3 else 'FAIL'}")
    ok &= err < 1e-3
    print("ALL OK" if ok else "FAILURES")
