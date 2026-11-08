"""2D konvolúció mérési sorozat: naiv globális vs. shared memory-os kernel.

A mérési protokoll a következőket követi:

  * JIT bemelegítés (a Numba fordítási ideje nem kerül a statisztikába),
  * a GPU kernel tiszta futásideje hardveres ``cuda.event`` méréssel,
  * a Host->Device és Device->Host másolás külön, host oldali stoppperrel,
  * ismételt futások mediánja és szórása,
  * numerikus ellenőrzés egy IEEE-754 float32 CPU referenciához
    (``@njit(fastmath=False)``, hogy a CPU baseline ne legyen lassabb a
    szükségesnél, de a lebegőpontos asszociativitás se változzon),
  * az eredmények CSV-be és ábrába írása.

Futtatás::

    python3 benchmark.py
"""

import csv
import math
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda, float32, njit
from scipy.ndimage import convolve

from conv2d import conv2d_global_kernel, conv2d_shared_kernel

SIZES = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
KERNEL_SIZES = [3, 5, 7, 15]
BLOCK = (16, 16)
GPU_RUNS = 20
CPU_RUNS = 3
COPY_RUNS = 3
TOLERANCE = 1e-3


@njit(fastmath=False)
def cpu_conv2d(image, kernel, out):
  """Szinkron, egy szálas float32 referencia (zero-padding, valódi konvolúció).

  A ``fastmath=False`` biztosítja, hogy a CPU a szigorú IEEE-754 szabvány
  szerint számoljon (nincs átrendezés), így a hibatolerancia értelmezhető.
  """
  h, w = image.shape
  k = kernel.shape[0]
  k_half = k // 2
  for y in range(h):
    for x in range(w):
      acc = float32(0.0)
      for ky in range(k):
        iy = y + ky - k_half
        if 0 <= iy < h:
          for kx in range(k):
            ix = x + kx - k_half
            if 0 <= ix < w:
              acc += image[iy, ix] * kernel[k - 1 - ky, k - 1 - kx]
      out[y, x] = acc


def verify_convolution_direction():
  """Aszimmetrikus magokkal ellenőrzi a konvolúció irányát (nem korreláció).

  Szimmetrikus mag mellett a konvolúció és a korreláció azonos eredményt ad,
  ezért a mérési sorozat előtt véletlen, aszimmetrikus magokkal hasonlítjuk
  össze mindkét GPU kernelt a SciPy ``convolve`` referencia implementációval.
  """
  rng = np.random.default_rng(1234)
  h, w = 37, 53
  image = (rng.random((h, w), dtype=np.float32) * 255.0)
  d_img = cuda.to_device(image)
  d_out = cuda.device_array((h, w), dtype=np.float32)
  grid = (math.ceil(w / BLOCK[0]), math.ceil(h / BLOCK[1]))
  all_ok = True
  print("Konvolúció-irány ellenőrzés aszimmetrikus magokkal (max |GPU - SciPy|):")
  for k in (3, 5, 7, 15):
    kernel = rng.random((k, k), dtype=np.float32) - 0.5
    ref = convolve(image, kernel, mode="constant", cval=0.0)
    d_k = cuda.to_device(kernel)
    tile_w = BLOCK[0] + k - 1

    conv2d_global_kernel[grid, BLOCK](d_img, d_k, d_out, w, h, k)
    cuda.synchronize()
    err_naive = float(np.max(np.abs(d_out.copy_to_host() - ref)))
    conv2d_shared_kernel[grid, BLOCK, 0, tile_w * tile_w * 4](
        d_img, d_k, d_out, w, h, k)
    cuda.synchronize()
    err_shared = float(np.max(np.abs(d_out.copy_to_host() - ref)))

    ok = err_naive < TOLERANCE and err_shared < TOLERANCE
    all_ok &= ok
    print(f"  k={k:2d}: naiv {err_naive:.2e} | shared {err_shared:.2e} "
          f"{'OK' if ok else 'HIBA'}")
  print()
  return all_ok


def gauss_kernel(k):
  """Normalizált Gauss-szerű, szeparálható szűrőmag (összege 1)."""
  half = k // 2
  x = np.arange(k, dtype=np.float64) - half
  sigma = max(k / 6.0, 0.5)
  g = np.exp(-(x * x) / (2.0 * sigma * sigma))
  kernel = np.outer(g, g)
  kernel /= kernel.sum()
  return kernel.astype(np.float32)


def time_cpu(image, kernel, out):
  cpu_conv2d(image, kernel, out)  # bemelegítés (fordítás)
  times = []
  for _ in range(CPU_RUNS):
    t0 = time.perf_counter()
    cpu_conv2d(image, kernel, out)
    times.append((time.perf_counter() - t0) * 1000.0)
  return statistics.median(times)


def time_gpu(kernel_fn, grid, block, args, shared_bytes=0):
  for _ in range(5):  # JIT bemelegítés
    kernel_fn[grid, block, 0, shared_bytes](*args)
  cuda.synchronize()

  times = []
  for _ in range(GPU_RUNS):
    start = cuda.event()
    end = cuda.event()
    start.record()
    kernel_fn[grid, block, 0, shared_bytes](*args)
    end.record()
    end.synchronize()
    times.append(start.elapsed_time(end))
  return statistics.median(times), statistics.pstdev(times)


def time_copy(copy_fn):
  times = []
  result = None
  for _ in range(COPY_RUNS):
    cuda.synchronize()
    t0 = time.perf_counter()
    result = copy_fn()
    cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000.0)
  return statistics.median(times), result


def main():
  if not cuda.is_available():
    raise SystemExit("Nincs elérhető CUDA eszköz.")

  if not verify_convolution_direction():
    raise SystemExit("A konvolúció-irány ellenőrzés hibát talált.")

  rng = np.random.default_rng(42)
  rows = []

  print("2D konvolúció: naiv vs. shared memory (Numba CUDA)")
  print(f"GPU: {cuda.get_current_device().name.decode()}")
  print(f"Blokk: {BLOCK[0]}x{BLOCK[1]} szál, kernel idő: {GPU_RUNS} futás "
        f"mediánja, CPU referencia: {CPU_RUNS} futás mediánja\n")
  header = (f"{'kép':>9} {'mag':>3} {'impl':>7} {'CPU [ms]':>9} "
            f"{'H2D [ms]':>8} {'kernel [ms]':>11} {'szórás':>7} "
            f"{'D2H [ms]':>8} {'max hiba':>9} {'gyorsítás':>9}")
  print(header)
  print("-" * len(header))

  for (height, width) in SIZES:
    image = (rng.random((height, width), dtype=np.float32) * 255.0)
    for k in KERNEL_SIZES:
      kernel = gauss_kernel(k)
      ref = np.zeros((height, width), dtype=np.float32)

      cpu_ms = time_cpu(image, kernel, ref)

      grid = (math.ceil(width / BLOCK[0]), math.ceil(height / BLOCK[1]))
      tile_w = BLOCK[0] + k - 1
      shared_bytes = tile_w * tile_w * 4

      h2d_ms, (d_img, d_kernel) = time_copy(
          lambda: (cuda.to_device(image), cuda.to_device(kernel)))
      d_out = cuda.device_array((height, width), dtype=np.float32)

      for impl, kernel_fn, smem in (
          ("naiv", conv2d_global_kernel, 0),
          ("shared", conv2d_shared_kernel, shared_bytes),
      ):
        kernel_ms, spread = time_gpu(
            kernel_fn, grid, BLOCK, (d_img, d_kernel, d_out, width, height, k),
            smem)
        d2h_ms, host_out = time_copy(d_out.copy_to_host)

        max_err = float(np.max(np.abs(host_out - ref)))
        allclose = bool(np.allclose(host_out, ref, atol=TOLERANCE))
        speedup = cpu_ms / kernel_ms
        total_ms = h2d_ms + kernel_ms + d2h_ms

        print(f"{height:>4}x{width:<4} {k:>3} {impl:>7} {cpu_ms:>9.2f} "
              f"{h2d_ms:>8.2f} {kernel_ms:>11.2f} {spread:>7.2f} "
              f"{d2h_ms:>8.2f} {max_err:>9.2e} {speedup:>8.1f}x"
              f"{'' if allclose else '  HIBA'}")

        rows.append({
            "height": height,
            "width": width,
            "k_size": k,
            "impl": impl,
            "cpu_ms": round(cpu_ms, 4),
            "h2d_ms": round(h2d_ms, 4),
            "kernel_ms": round(kernel_ms, 4),
            "kernel_std_ms": round(spread, 4),
            "d2h_ms": round(d2h_ms, 4),
            "total_gpu_ms": round(total_ms, 4),
            "max_abs_err": f"{max_err:.3e}",
            "allclose": allclose,
            "speedup_vs_cpu": round(speedup, 3),
        })

  with open("benchmark_2d.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print("\nEredmények: benchmark_2d.csv")

  # ---------------------------------------------------------------------
  # Ábra: kernel idő és gyorsulás a képméret függvényében
  # ---------------------------------------------------------------------
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
  colors = {3: "tab:blue", 5: "tab:orange", 7: "tab:green", 15: "tab:red"}
  for k in KERNEL_SIZES:
    for impl, style in (("naiv", "--"), ("shared", "-")):
      pts = [r for r in rows if r["k_size"] == k and r["impl"] == impl]
      pixels = [r["height"] * r["width"] for r in pts]
      ms = [r["kernel_ms"] for r in pts]
      axes[0].plot(pixels, ms, style, color=colors[k], marker="o",
                   label=f"k={k} {impl}")
      axes[1].plot(pixels, [r["speedup_vs_cpu"] for r in pts], style,
                   color=colors[k], marker="o", label=f"k={k} {impl}")
  axes[0].set_xscale("log", base=2)
  axes[0].set_yscale("log")
  axes[0].set_xlabel("kép elemszáma (pixel)")
  axes[0].set_ylabel("kernel futásidő [ms]")
  axes[0].set_title("GPU kernel idő")
  axes[0].grid(True, which="both", alpha=0.3)
  axes[1].set_xscale("log", base=2)
  axes[1].set_xlabel("kép elemszáma (pixel)")
  axes[1].set_ylabel("gyorsulás a CPU referenciahoz")
  axes[1].set_title("Speedup (kernel idő)")
  axes[1].grid(True, which="both", alpha=0.3)
  axes[0].legend(fontsize=7, ncol=2)
  fig.tight_layout()
  fig.savefig("benchmark_2d.png", dpi=140)
  print("Ábra: benchmark_2d.png")


if __name__ == "__main__":
  main()
