import numpy as np
from numba import cuda
import math

@cuda.jit
def naive_linear_regression_kernel(x, y, N, out):
    """
    Naiv GPU kernel: Nincs shared memory redukció.
    Minden szál grid-stride ciklussal feldolgozza a rábízott elemeket,
    majd közvetlenül a globális memóriába végzi az atomi összeadást.
    
    out tömb:
      out[0]: sum(x)
      out[1]: sum(y)
      out[2]: sum(x^2)
      out[3]: sum(x*y)
    """
    gid = cuda.grid(1)
    stride = cuda.gridsize(1)

    # Lokális akkumulátorok a szál regisztereiben
    local_sum_x  = 0.0
    local_sum_y  = 0.0
    local_sum_xx = 0.0
    local_sum_xy = 0.0

    # Grid-stride ciklus
    for i in range(gid, N, stride):
        xi = x[i]
        yi = y[i]
        local_sum_x  += xi
        local_sum_y  += yi
        local_sum_xx += xi * xi
        local_sum_xy += xi * yi

    # Közvetlen írás a globális memóriába atomi művelettel
    # (Ez nagy elemszámnál jelentős sávszélesség-szűkességhez vezet)
    if N > 0:  # Csak akkor írunk, ha a szál kapott munkát
        cuda.atomic.add(out, 0, local_sum_x)
        cuda.atomic.add(out, 1, local_sum_y)
        cuda.atomic.add(out, 2, local_sum_xx)
        cuda.atomic.add(out, 3, local_sum_xy)


def fit_linear_regression_naive_gpu(x, y):
    """
    Host függvény a naiv GPU kernel indításához.
    """
    N = x.shape[0]
    
    # Adatok átvitele a GPU-ra
    d_x = cuda.to_device(np.ascontiguousarray(x, dtype=np.float64))
    d_y = cuda.to_device(np.ascontiguousarray(y, dtype=np.float64))
    d_out = cuda.to_device(np.zeros(4, dtype=np.float64))

    threads_per_block = 256
    blocks_per_grid = min((N + threads_per_block - 1) // threads_per_block, 1024)

    # Naiv kernel indítása
    naive_linear_regression_kernel[blocks_per_grid, threads_per_block](d_x, d_y, N, d_out)

    # Eredmények visszahozatala
    h_out = d_out.copy_to_host()
    sum_x, sum_y, sum_xx, sum_xy = h_out[0], h_out[1], h_out[2], h_out[3]

    # Egyenes együtthatóinak kiszámítása a CPU-n
    denominator = N * sum_xx - sum_x * sum_x
    if math.isclose(denominator, 0.0):
        raise ValueError("A nevező nulla vagy túl közel van a nullához.")

    beta_1 = (N * sum_xy - sum_x * sum_y) / denominator
    beta_0 = (sum_y - beta_1 * sum_x) / N

    return beta_1, beta_0