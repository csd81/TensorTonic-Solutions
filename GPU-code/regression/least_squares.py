import numpy as np
from numba import cuda
import math

# Blokk méret definiálása fordítási konstansként (2 hatványa ajánlott)
THREADS_PER_BLOCK = 256

@cuda.jit
def linear_regression_kernel(x, y, N, out):
    """
    out tömb tartalma:
      out[0]: sum(x)
      out[1]: sum(y)
      out[2]: sum(x^2)
      out[3]: sum(x*y)
    """
    # 1. Shared memory allokálása a 4 redukciós célra blokkonként
    s_x  = cuda.shared.array(THREADS_PER_BLOCK, dtype=np.float64)
    s_y  = cuda.shared.array(THREADS_PER_BLOCK, dtype=np.float64)
    s_xx = cuda.shared.array(THREADS_PER_BLOCK, dtype=np.float64)
    s_xy = cuda.shared.array(THREADS_PER_BLOCK, dtype=np.float64)

    tid = cuda.threadIdx.x
    gid = cuda.grid(1)
    stride = cuda.gridsize(1)

    # 2. Lokális regiszter-akkumulátorok inicializálása
    local_sum_x  = 0.0
    local_sum_y  = 0.0
    local_sum_xx = 0.0
    local_sum_xy = 0.0

    # 3. Grid-stride ciklus: minden szál több elemet is összegezhet, ha N > szálak száma
    for i in range(gid, N, stride):
        xi = x[i]
        yi = y[i]
        local_sum_x  += xi
        local_sum_y  += yi
        local_sum_xx += xi * xi
        local_sum_xy += xi * yi

    # Eredmények betöltése a blokk megosztott memóriájába
    s_x[tid]  = local_sum_x
    s_y[tid]  = local_sum_y
    s_xx[tid] = local_sum_xx
    s_xy[tid] = local_sum_xy

    cuda.syncthreads()

    # 4. Párhuzamos bináris fa-redukció a shared memory-ban
    # Felező léptékkel haladunk, elkerülve az elágazási divergenciát
    stride_step = THREADS_PER_BLOCK // 2
    while stride_step > 0:
        if tid < stride_step:
            s_x[tid]  += s_x[tid + stride_step]
            s_y[tid]  += s_y[tid + stride_step]
            s_xx[tid] += s_xx[tid + stride_step]
            s_xy[tid] += s_xy[tid + stride_step]
        cuda.syncthreads()
        stride_step //= 2

    # 5. Blokkonkénti összesítés: csak a 0-s indexű szál ír atomi művelettel a globális memóriába
    if tid == 0:
        cuda.atomic.add(out, 0, s_x[0])
        cuda.atomic.add(out, 1, s_y[0])
        cuda.atomic.add(out, 2, s_xx[0])
        cuda.atomic.add(out, 3, s_xy[0])


def fit_linear_regression_gpu(x, y):
    """
    Futtató és paraméterszámító host függvény.
    """
    N = x.shape[0]
    
    # Bemeneti adatok átvitele a GPU-ra (64 bites lebegőpont a numerikus stabilitásért)
    d_x = cuda.to_device(np.ascontiguousarray(x, dtype=np.float64))
    d_y = cuda.to_device(np.ascontiguousarray(y, dtype=np.float64))
    
    # Kimeneti tömb inicializálása a GPU-n nullákkal
    d_out = cuda.to_device(np.zeros(4, dtype=np.float64))

    # Grid és blokk dimenziók beállítása
    threads_per_block = THREADS_PER_BLOCK
    blocks_per_grid = min((N + threads_per_block - 1) // threads_per_block, 1024)

    # Kernel indítása
    linear_regression_kernel[blocks_per_grid, threads_per_block](d_x, d_y, N, d_out)

    # Részösszegek visszahozatása a CPU-ra
    h_out = d_out.copy_to_host()
    sum_x, sum_y, sum_xx, sum_xy = h_out[0], h_out[1], h_out[2], h_out[3]

    # Skalár számítás a CPU-n
    denominator = N * sum_xx - sum_x * sum_x
    if math.isclose(denominator, 0.0):
        raise ValueError("A nevező nulla vagy túl közel van a nullához (az x értékek azonosak).")

    beta_1 = (N * sum_xy - sum_x * sum_y) / denominator
    beta_0 = (sum_y - beta_1 * sum_x) / N

    return beta_1, beta_0