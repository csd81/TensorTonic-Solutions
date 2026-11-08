"""Numba CUDA port of color_inversion.cu.

In-place RGBA color inversion: out.rgb = 255 - in.rgb, alpha untouched.

Because 255 - c == c ^ 0xFF for a byte, a whole pixel is inverted by XORing
its packed 32-bit value with 0x00FFFFFF (alpha is the high byte).  The image
is viewed as a uint32 array, so each thread handles one pixel with a single
32-bit load and store instead of six byte accesses.

solve(image, width, height) keeps the challenge signature and modifies the
image in place.  Both Numba device arrays and host NumPy arrays are accepted
(host arrays are copied to the device and back by Numba).
"""

import numpy as np
from numba import cuda

XOR_MASK = np.uint32(0x00FFFFFF)
THREADS = 256


@cuda.jit
def invert_kernel(image, n):
    i = cuda.grid(1)
    if i < n:
        image[i] ^= XOR_MASK


def solve(image, width, height):
    if image is None or width <= 0 or height <= 0:
        return

    pixels = width * height
    img = image
    if getattr(img, "ndim", 1) != 1:
        img = img.reshape(-1)
    img32 = img.view(np.uint32)

    blocks = (pixels + THREADS - 1) // THREADS
    invert_kernel[blocks, THREADS](img32, pixels)
    cuda.synchronize()


def _check(name, data, width, height):
    """Runs solve in place and compares with the reference inversion."""
    original = data.copy()
    solve(data, width, height)

    pixels = width * height
    ref = original.reshape(pixels, 4).copy()
    ref[:, :3] = 255 - ref[:, :3]
    got = data.reshape(pixels, 4)
    bad = int(np.count_nonzero(got != ref))
    rel = bad / float(got.size)
    print(f"{name:>20} w={width} h={height} px={pixels}: "
          f"{'OK' if bad == 0 else 'FAIL'} (mismatches={bad}, {rel:.1e})")
    return bad == 0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)

    # Challenge examples (host NumPy arrays, in place).
    ok = True
    ex1 = np.array([255, 0, 128, 255, 0, 255, 0, 255], dtype=np.uint8)
    expected1 = np.array([0, 255, 127, 255, 255, 0, 255, 255], dtype=np.uint8)
    solve(ex1, 1, 2)
    print("example 1:", ex1.tolist(), "OK" if np.array_equal(ex1, expected1)
          else "FAIL")
    ok &= np.array_equal(ex1, expected1)

    ex2 = np.array([10, 20, 30, 255, 100, 150, 200, 255], dtype=np.uint8)
    expected2 = np.array([245, 235, 225, 255, 155, 105, 55, 255],
                         dtype=np.uint8)
    solve(ex2, 2, 1)
    print("example 2:", ex2.tolist(), "OK" if np.array_equal(ex2, expected2)
          else "FAIL")
    ok &= np.array_equal(ex2, expected2)

    # Random images through host arrays and device arrays.
    for w, h in [(1, 1), (3, 5), (7, 3), (100, 37), (1024, 1), (1, 1024),
                 (12345, 1), (257, 131), (512, 512)]:
        data = rng.integers(0, 256, (w * h * 4,), dtype=np.uint8)
        ok &= _check("host random", data, w, h)

        dev = cuda.to_device(data.copy())
        original = data.copy()
        solve(dev, w, h)
        got = dev.copy_to_host()
        ref = original.reshape(-1, 4).copy()
        ref[:, :3] = 255 - ref[:, :3]
        ok &= bool(np.array_equal(got, ref.reshape(-1)))
        print(f"{'device random':>20} w={w} h={h}: "
              f"{'OK' if np.array_equal(got, ref.reshape(-1)) else 'FAIL'}")

    # Performance: 4096x5120 (the challenge measurement size), median of 20.
    import statistics
    import time

    width, height = 4096, 5120
    n = width * height * 4
    data = rng.integers(0, 256, (n,), dtype=np.uint8)
    dev = cuda.to_device(data)
    solve(dev, width, height)  # warm-up
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        solve(dev, width, height)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    hw_gbps = 2.0 * n / (ms * 1e6)          # full read + write traffic
    rgb_gbps = 1.75 * n / (ms * 1e6)        # 4 channels read, 3 written
    print(f"Perf {width}x{height} ({n} bytes): {ms:.3f} ms, "
          f"~{hw_gbps:.1f} GB/s hardware / ~{rgb_gbps:.1f} GB/s RGB traffic")
