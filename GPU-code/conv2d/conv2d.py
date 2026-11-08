import math
import statistics
import time
import numpy as np
from numba import cuda, float32
from scipy.ndimage import convolve

# ==============================================================================
# 1. 2D CUDA KERNEL
# ==============================================================================


@cuda.jit
def conv2d_global_kernel(img_in, kernel, img_out, width, height, k_size):
  """Naiv/Globális memóriás 2D konvolúciós kernel:

  Minden szál egy kimeneti pixel értékét számolja ki a 2D rács mentén.
  """
  # 2D szálkoordináták
  tx = cuda.threadIdx.x
  ty = cuda.threadIdx.y
  bx = cuda.blockIdx.x
  by = cuda.blockIdx.y
  bw = cuda.blockDim.x
  bh = cuda.blockDim.y

  col = bx * bw + tx
  row = by * bh + ty

  k_half = k_size // 2

  if col < width and row < height:
    acc = float32(0.0)
    for ky in range(k_size):
      for kx in range(k_size):
        img_x = col + (kx - k_half)
        img_y = row + (ky - k_half)

        # Zero-padding peremkezelés
        if 0 <= img_x < width and 0 <= img_y < height:
          # Valódi konvolúció: a szűrőmagot megfordítva olvassuk
          acc += img_in[img_y, img_x] * kernel[k_size - 1 - ky, k_size - 1 - kx]

    img_out[row, col] = acc


# ==============================================================================
# 2. TESZTFUTTATÁS ÉS TELJESÍTMÉNYMÉRÉS
# ==============================================================================


def run_2d_benchmark():
  # Kép- és szűrőméretek beállítása
  WIDTH, HEIGHT = 4096, 4096  # 16 Megapixeles képmátrix
  K_SIZE = 5  # 5x5-ös szűrőmag (pl. Gauss simítás)

  if K_SIZE < 1 or K_SIZE % 2 == 0:
    raise ValueError(
        f"A szűrőmag mérete páratlan pozitív szám legyen (a SciPy konvolúció"
        f" origója miatt), kapott érték: {K_SIZE}"
    )

  print(f"--- 2D Képi Konvolúciós Benchmark ---")
  print(f"Kép mérete: {WIDTH} x {HEIGHT} pixel ({WIDTH * HEIGHT / 1e6:.1f} MP)")
  print(f"Szűrőmag mérete: {K_SIZE}x{K_SIZE}\n")

  # Bemeneti adatok generálása (float32)
  np.random.seed(42)
  image_host = np.random.rand(HEIGHT, WIDTH).astype(np.float32) * 255.0

  # 5x5 Gauss-szerű szűrőmag generálása
  kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32)
  kernel_host = np.outer(kernel_1d, kernel_1d)
  kernel_host /= kernel_host.sum()  # Normalizálás

  # --------------------------------------------------------------------------
  # 0. Kis, aszimmetrikus maggal ellenőrizzük a valódi konvolúciót
  #    (szimmetrikus mag mellett a konvolúció és korreláció azonos lenne)
  # --------------------------------------------------------------------------
  test_img = np.arange(25, dtype=np.float32).reshape(5, 5)
  test_kernel = np.arange(9, dtype=np.float32).reshape(3, 3)
  test_ref = convolve(test_img, test_kernel, mode="constant", cval=0.0)

  d_test_img = cuda.to_device(test_img)
  d_test_kernel = cuda.to_device(test_kernel)
  d_test_out = cuda.device_array((5, 5), dtype=np.float32)
  conv2d_global_kernel[(1, 1), (16, 16)](
      d_test_img, d_test_kernel, d_test_out, 5, 5, 3
  )
  cuda.synchronize()
  test_err = float(np.max(np.abs(d_test_out.copy_to_host() - test_ref)))
  print(
      f"Konvolúció-ellenőrzés aszimmetrikus 3x3 maggal:"
      f" max hiba = {test_err:.2e} ({'OK' if test_err < 1e-5 else 'HIBA'})\n"
  )

  # --------------------------------------------------------------------------
  # 1. CPU Referencia (SciPy convolve)
  # --------------------------------------------------------------------------
  print("CPU referencia futtatása (SciPy)...")
  cpu_times = []
  image_cpu_ref = None
  for _ in range(3):
    start_cpu = time.perf_counter()
    # mode='constant', cval=0.0 felel meg a zero-paddingnek
    image_cpu_ref = convolve(
        image_host, kernel_host, mode="constant", cval=0.0
    )
    cpu_times.append((time.perf_counter() - start_cpu) * 1000.0)
  time_cpu = statistics.median(cpu_times)
  print(f"CPU futásidő: {time_cpu:.2f} ms (3 futás mediánja)")

  # --------------------------------------------------------------------------
  # 2. GPU Konfiguráció és Adatmozgatás
  # --------------------------------------------------------------------------
  cuda.synchronize()
  h2d_start = time.perf_counter()
  d_image_in = cuda.to_device(image_host)
  d_kernel = cuda.to_device(kernel_host)
  cuda.synchronize()
  time_h2d = (time.perf_counter() - h2d_start) * 1000.0
  print(f"Host -> Device másolás: {time_h2d:.2f} ms")

  d_image_out = cuda.device_array((HEIGHT, WIDTH), dtype=np.float32)

  # 2D Blokkméret definíció (pl. 16x16 = 256 szál blokkonként)
  blockdim = (16, 16)
  griddim = (
      math.ceil(WIDTH / blockdim[0]),
      math.ceil(HEIGHT / blockdim[1]),
  )

  # JIT bemelegítés (Warm-up)
  for _ in range(5):
    conv2d_global_kernel[griddim, blockdim](
        d_image_in, d_kernel, d_image_out, WIDTH, HEIGHT, K_SIZE
    )
  cuda.synchronize()

  # --------------------------------------------------------------------------
  # 3. GPU Teljesítménymérés (cuda.event, medián)
  # --------------------------------------------------------------------------
  gpu_times = []
  for _ in range(20):
    start_event = cuda.event()
    end_event = cuda.event()
    start_event.record()
    conv2d_global_kernel[griddim, blockdim](
        d_image_in, d_kernel, d_image_out, WIDTH, HEIGHT, K_SIZE
    )
    end_event.record()
    end_event.synchronize()
    gpu_times.append(start_event.elapsed_time(end_event))
  time_gpu = statistics.median(gpu_times)

  # Eredmény letöltése és ellenőrzése
  d2h_start = time.perf_counter()
  image_gpu_res = d_image_out.copy_to_host()
  cuda.synchronize()
  time_d2h = (time.perf_counter() - d2h_start) * 1000.0

  max_err = float(np.max(np.abs(image_cpu_ref - image_gpu_res)))
  correct = np.allclose(image_cpu_ref, image_gpu_res, atol=1e-3)

  print(f"GPU futásidő:   {time_gpu:.2f} ms (20 futás mediánja)")
  print(f"Device -> Host másolás: {time_d2h:.2f} ms")
  print(f"Max abszolút hiba: {max_err:.3e}")
  print(f"Numerikus helyesség: {'MEGFELEL (OK)'  if correct else 'HIBA'}")
  print(f"Gyorsulás (Speedup): {time_cpu / time_gpu:.1f}x")


if __name__ == "__main__":
  run_2d_benchmark()