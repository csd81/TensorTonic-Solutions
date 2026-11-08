import math
import statistics
import time
import numpy as np
from numba import cuda, float32

# ==============================================================================
# 1. GPU KERNELEK (NUMBA CUDA)
# ==============================================================================

# Blokkméret és maximális szűrőhossz definiálása
TPB_1D = 256
MAX_KERNEL_RADIUS = 64  # Legfeljebb 129 pontos szűrőmaghoz (2 * 64 + 1)
SHARED_SIZE = TPB_1D + 2 * MAX_KERNEL_RADIUS


@cuda.jit
def conv1d_naive_kernel(x, h, y, n_samples, n_kernel):
    """Naiv 1D konvolúciós kernel:

    Minden szál a globális memóriából olvassa a szomszédos mintákat.
    Peremkezelés: zero-padding.
    """
    idx = cuda.grid(1)
    # A NumPy mode="same" középpontja: (K - 1) // 2 (páros K-ra is ez a helyes)
    k_half = (n_kernel - 1) // 2

    if idx < n_samples:
        acc = float32(0.0)
        for k in range(n_kernel):
            # Konvolúció: h megfordított indexelése
            sample_idx = idx - (k - k_half)

            # Zero-padding ellenőrzés
            if 0 <= sample_idx < n_samples:
                acc += x[sample_idx] * h[k]

        y[idx] = acc


@cuda.jit
def conv1d_shared_kernel(x, h, y, n_samples, n_kernel):
    """Optimalizált 1D konvolúció megosztott memóriával (Shared Memory Tiling):

    A szálblokk betölti a belső mintákat és a szélső halo/apron cellákat is,
    így a belső ciklus a gyors Shared Memory-ból fut.
    """
    # Statikus méretű megosztott memória lefoglalása
    # Mérete: Blokkméret + bal oldali halo + jobb oldali halo
    shared_mem = cuda.shared.array(SHARED_SIZE, float32)

    tid = cuda.threadIdx.x
    gid = cuda.grid(1)
    k_half = (n_kernel - 1) // 2
    halo_size = n_kernel // 2

    # 1. Belső elemek betöltése a shared memóriába
    shared_idx = tid + halo_size
    if gid < n_samples:
        shared_mem[shared_idx] = x[gid]
    else:
        shared_mem[shared_idx] = float32(0.0)

    # 2. Bal oldali perem (halo) betöltése a blokk első 'halo_size' szála által
    if halo_size > 0 and tid < halo_size:
        left_gid = gid - halo_size
        if left_gid >= 0 and left_gid < n_samples:
            shared_mem[tid] = x[left_gid]
        else:
            shared_mem[tid] = float32(0.0)

    # 3. Jobb oldali perem (halo) betöltése a blokk utolsó 'halo_size' szála által
    if halo_size > 0 and tid >= cuda.blockDim.x - halo_size:
        right_gid = gid + halo_size
        target_idx = shared_idx + halo_size
        if right_gid < n_samples:
            shared_mem[target_idx] = x[right_gid]
        else:
            shared_mem[target_idx] = float32(0.0)

    # Szinkronizáció: megvárjuk, míg a teljes blokk befejezi a betöltést
    cuda.syncthreads()

    # 4. Konvolúció számítása a megosztott memóriából
    if gid < n_samples:
        acc = float32(0.0)
        for k in range(n_kernel):
            s_idx = shared_idx - (k - k_half)
            acc += shared_mem[s_idx] * h[k]
        y[gid] = acc


# ==============================================================================
# 2. TESZTFUTTATÁS ÉS TELJESÍTMÉNYMÉRÉS
# ==============================================================================


def run_benchmark():
    # Bemeneti méretek beállítása
    N_SAMPLES = 5_000_000  # 5 millió adatpont
    KERNEL_SIZE = 31  # 31 pontos FIR aluláteresztő szűrőmag
    THREADS_PER_BLOCK = TPB_1D
    BLOCKS = math.ceil(N_SAMPLES / THREADS_PER_BLOCK)

    if not 1 <= KERNEL_SIZE <= 2 * MAX_KERNEL_RADIUS + 1:
        raise ValueError(
            f"A szűrőmag hossza 1 és {2 * MAX_KERNEL_RADIUS + 1} között legyen,"
            f" kapott érték: {KERNEL_SIZE}"
        )

    print(f"--- 1D Konvolúciós Benchmark ---")
    print(f"Jel hossza: {N_SAMPLES:,} minta")
    print(f"Szűrőmag mérete: {KERNEL_SIZE} tap")
    print(f"Grid elrendezés: {BLOCKS} blokk x {THREADS_PER_BLOCK} szál\n")

    # Bemeneti adatok generálása (float32 precízióval)
    np.random.seed(42)
    x_host = np.random.randn(N_SAMPLES).astype(np.float32)
    # Normált egyszerű szűrőmag (pl. simító szűrő)
    h_host = (np.ones(KERNEL_SIZE) / KERNEL_SIZE).astype(np.float32)

    # --------------------------------------------------------------------------
    # CPU Referencia (NumPy convolve - mode='same')
    # --------------------------------------------------------------------------
    cpu_times = []
    y_cpu = None
    for _ in range(3):
        start_cpu = time.perf_counter()
        y_cpu = np.convolve(x_host, h_host, mode="same")
        cpu_times.append((time.perf_counter() - start_cpu) * 1000.0)
    time_cpu = statistics.median(cpu_times)
    print(f"NumPy (CPU) futásidő: {time_cpu:.2f} ms (3 futás mediánja)")

    # --------------------------------------------------------------------------
    # GPU Adatmozgatás (Host -> Device)
    # --------------------------------------------------------------------------
    cuda.synchronize()
    h2d_start = time.perf_counter()
    d_x = cuda.to_device(x_host)
    d_h = cuda.to_device(h_host)
    cuda.synchronize()
    time_h2d = (time.perf_counter() - h2d_start) * 1000.0
    print(f"Host -> Device másolás: {time_h2d:.2f} ms")

    d_y_naive = cuda.device_array(N_SAMPLES, dtype=np.float32)
    d_y_shared = cuda.device_array(N_SAMPLES, dtype=np.float32)

    def time_kernel(kernel, d_out, repeats=20):
        """Warm-up + ismételt mérés, mediánnal."""
        for _ in range(5):
            kernel[BLOCKS, THREADS_PER_BLOCK](
                d_x, d_h, d_out, N_SAMPLES, KERNEL_SIZE
            )
        cuda.synchronize()

        times = []
        for _ in range(repeats):
            start_event = cuda.event()
            end_event = cuda.event()
            start_event.record()
            kernel[BLOCKS, THREADS_PER_BLOCK](
                d_x, d_h, d_out, N_SAMPLES, KERNEL_SIZE
            )
            end_event.record()
            end_event.synchronize()
            times.append(start_event.elapsed_time(end_event))
        return statistics.median(times)

    # --------------------------------------------------------------------------
    # GPU Naiv Kernel mérése
    # --------------------------------------------------------------------------
    time_naive = time_kernel(conv1d_naive_kernel, d_y_naive)

    d2h_start = time.perf_counter()
    y_gpu_naive = d_y_naive.copy_to_host()
    cuda.synchronize()
    time_d2h_naive = (time.perf_counter() - d2h_start) * 1000.0
    correct_naive = np.allclose(y_cpu, y_gpu_naive, atol=1e-4)

    print(
        f"Naiv GPU futásidő:     {time_naive:.2f} ms (D2H: {time_d2h_naive:.2f} ms)"
        f" | Pontosság: {'OK' if correct_naive else 'HIBA'}"
        f" | Gyorsulás: {time_cpu / time_naive:.1f}x"
    )

    # --------------------------------------------------------------------------
    # GPU Shared Memory Kernel mérése
    # --------------------------------------------------------------------------
    time_shared = time_kernel(conv1d_shared_kernel, d_y_shared)

    d2h_start = time.perf_counter()
    y_gpu_shared = d_y_shared.copy_to_host()
    cuda.synchronize()
    time_d2h_shared = (time.perf_counter() - d2h_start) * 1000.0
    correct_shared = np.allclose(y_cpu, y_gpu_shared, atol=1e-4)

    print(
        f"Shared GPU futásidő:   {time_shared:.2f} ms (D2H: {time_d2h_shared:.2f} ms)"
        f" | Pontosság: {'OK' if correct_shared else 'HIBA'}"
        f" | Gyorsulás: {time_cpu / time_shared:.1f}x"
    )


if __name__ == "__main__":
    run_benchmark()