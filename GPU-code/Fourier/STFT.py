import math
import time
import numpy as np
from scipy import signal as sp_signal
from numba import cuda, float32

MAX_FFT = 1024


# ---------------------------------------------------------------------------
# 1. Numba CUDA STFT Kernel (Shared Memory és Coalesced olvasás)
# ---------------------------------------------------------------------------
@cuda.jit
def stft_kernel_shared(d_signal, d_window, d_real_out, d_imag_out,
                       signal_len, frame_len, hop_size, n_frames, n_fft,
                       scale):
    """
    Minden Thread Block egy időkeretet (frame) dolgoz fel.
    A blokkon belüli szálak felelősek a shared memory betöltéséért
    és a valós DFT frekvenciabin-összegekért (RFFT: 0 .. n_fft//2).

    A kimenetet a SciPy stft(scaling="spectrum") konvenciójához igazítjuk:
    a nyers DFT-t 1 / sum(window) tényezővel skálázzuk.
    """
    # 1024 minta maximális méretig lefoglalt blokkon belüli shared memória
    s_frame = cuda.shared.array(MAX_FFT, float32)

    frame_idx = cuda.blockIdx.x
    tx = cuda.threadIdx.x

    if frame_idx >= n_frames:
        return

    frame_start = frame_idx * hop_size

    # 1. Adatbetöltés és ablakozás: Coalesced elérés a globális memóriából
    idx = tx
    while idx < n_fft:
        sig_pos = frame_start + idx
        if idx < frame_len and sig_pos < signal_len:
            s_frame[idx] = d_signal[sig_pos] * d_window[idx]
        else:
            s_frame[idx] = 0.0  # Zero-padding
        idx += cuda.blockDim.x

    cuda.syncthreads()

    # 2. RFFT összegzés: csak a pozitív frekvenciák (n_fft // 2 + 1 darab)
    n_freq_bins = (n_fft // 2) + 1
    m = tx
    while m < n_freq_bins:
        sum_real = 0.0
        sum_imag = 0.0
        angle_step = -2.0 * math.pi * m / n_fft
        cos_step = math.cos(angle_step)
        sin_step = math.sin(angle_step)
        c = 1.0
        s = 0.0

        for n in range(n_fft):
            val = s_frame[n]
            sum_real += val * c
            sum_imag += val * s
            # Trigonometrikus rekurzió: nincs cos/sin hívás a belső ciklusban
            c_new = c * cos_step - s * sin_step
            s_new = s * cos_step + c * sin_step
            c = c_new
            s = s_new

        # Transzponált elrendezés: [frekvencia_bin, idő_keret] mint a SciPy-ban
        d_real_out[m, frame_idx] = sum_real * scale
        d_imag_out[m, frame_idx] = sum_imag * scale

        m += cuda.blockDim.x


# ---------------------------------------------------------------------------
# 2. Host wrapper függvény
# ---------------------------------------------------------------------------
def run_numba_stft(x, window, hop_size, n_fft, measure_gpu_events=False):
    x = np.asarray(x)
    window = np.asarray(window)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if window.ndim != 1:
        raise ValueError(f"window must be 1-D, got shape {window.shape}")

    signal_len = x.shape[0]
    frame_len = window.shape[0]
    if frame_len < 1:
        raise ValueError("window must not be empty")
    if not frame_len <= n_fft <= MAX_FFT:
        raise ValueError(
            f"n_fft must satisfy frame_len <= n_fft <= {MAX_FFT}, "
            f"got frame_len={frame_len}, n_fft={n_fft}"
        )
    if not 1 <= hop_size <= frame_len:
        raise ValueError(
            f"hop_size must satisfy 1 <= hop_size <= frame_len, got {hop_size}"
        )
    if signal_len < frame_len:
        raise ValueError("signal is shorter than the window")

    n_frames = (signal_len - frame_len) // hop_size + 1
    n_freq_bins = (n_fft // 2) + 1
    threads_per_block = min(256, n_fft)
    blocks_per_grid = n_frames

    # SciPy stft(scaling="spectrum") konvenció: 1 / sum(window)
    win32 = np.ascontiguousarray(window, dtype=np.float32)
    scale = float(1.0 / np.sum(win32.astype(np.float64)))

    # Memória előkészítés
    d_signal = cuda.to_device(np.ascontiguousarray(x, dtype=np.float32))
    d_window = cuda.to_device(win32)
    d_real = cuda.device_array((n_freq_bins, n_frames), dtype=np.float32)
    d_imag = cuda.device_array((n_freq_bins, n_frames), dtype=np.float32)

    if measure_gpu_events:
        start_evt = cuda.event()
        stop_evt = cuda.event()
        start_evt.record()

    # Kernel hívás
    stft_kernel_shared[blocks_per_grid, threads_per_block](
        d_signal, d_window, d_real, d_imag,
        signal_len, frame_len, hop_size, n_frames, n_fft, scale
    )

    if measure_gpu_events:
        stop_evt.record()
        stop_evt.synchronize()
        kernel_time_ms = cuda.event_elapsed_time(start_evt, stop_evt)
    else:
        cuda.synchronize()
        kernel_time_ms = None

    h_real = d_real.copy_to_host()
    h_imag = d_imag.copy_to_host()

    gpu_stft_result = h_real + 1j * h_imag
    return gpu_stft_result, kernel_time_ms


# ---------------------------------------------------------------------------
# 3. Numerikus verifikáció (Összehasonlítás a SciPy referenciával)
# ---------------------------------------------------------------------------
def verify_correctness(x, window_name, frame_len, hop_size, n_fft):
    print("=== NUMERIKUS VERIFIKÁCIÓ ===")
    win = sp_signal.get_window(window_name, frame_len, fftbins=True).astype(np.float32)

    # Numba STFT futtatása (JIT bemelegítéssel)
    run_numba_stft(x[:1000], win, hop_size, n_fft)  # Bemelegítés
    gpu_out, _ = run_numba_stft(x, win, hop_size, n_fft)

    # SciPy STFT referencia hívás
    # boundary=None és padded=False biztosítja, hogy a SciPy pontosan a megadott kereteken dolgozzon
    _, _, scipy_out = sp_signal.stft(
        x,
        window=win,
        nperseg=frame_len,
        noverlap=frame_len - hop_size,
        nfft=n_fft,
        boundary=None,
        padded=False
    )

    # Dimenziók egyezőségének ellenőrzése
    min_frames = min(gpu_out.shape[1], scipy_out.shape[1])
    gpu_out = gpu_out[:, :min_frames]
    scipy_out = scipy_out[:, :min_frames]

    # Hiba metrikák
    diff = np.abs(gpu_out - scipy_out)
    max_err = np.max(diff)
    frobenius_err = np.linalg.norm(diff) / np.linalg.norm(scipy_out)
    is_close = np.allclose(gpu_out, scipy_out, atol=1e-4, rtol=1e-4)

    print(f"Bemeneti méret:             {x.size:,} minta")
    print(f"Kimeneti mátrix mérete:     {gpu_out.shape} (frekvenciabin x időkeret)")
    print(f"Maximális abszolút hiba:    {max_err:.6e}")
    print(f"Relatív Frobenius-hiba:     {frobenius_err:.6e}")
    print(f"Tolerancián belüli egyezés: {'IGEN (SIKERES)' if is_close else 'NEM'}\n")
    return is_close


# ---------------------------------------------------------------------------
# 4. Teljesítménymérés (Benchmark)
# ---------------------------------------------------------------------------
def benchmark_stft(sample_rates_seconds=[2, 5, 10], sample_rate=44100, n_fft=512, hop_size=256):
    print("=== TELJESÍTMÉNYMÉRÉS (BENCHMARK) ===")
    frame_len = n_fft
    win = sp_signal.get_window('hann', frame_len, fftbins=True).astype(np.float32)

    header = f"{'Hossz (mp)':<12}{'Minták':<12}{'SciPy (ms)':<14}{'GPU Kernel (ms)':<18}{'GPU Total (ms)':<16}{'Kernel Gyorsulás':<16}"
    print(header)
    print("-" * len(header))

    for sec in sample_rates_seconds:
        total_samples = int(sec * sample_rate)
        x = np.random.randn(total_samples).astype(np.float32)

        # 1. SciPy CPU mérés (átlag 3 futásból)
        scipy_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = sp_signal.stft(
                x, window=win, nperseg=frame_len,
                noverlap=frame_len - hop_size, nfft=n_fft,
                boundary=None, padded=False
            )
            scipy_times.append((time.perf_counter() - t0) * 1000.0)
        scipy_ms = np.median(scipy_times)

        # 2. Numba GPU mérés (kernel és total H2D/D2H transzferrel)
        total_times = []
        kernel_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            _, k_ms = run_numba_stft(x, win, hop_size, n_fft, measure_gpu_events=True)
            cuda.synchronize()
            total_times.append((time.perf_counter() - t0) * 1000.0)
            kernel_times.append(k_ms)

        total_ms = np.median(total_times)
        kernel_ms = np.median(kernel_times)
        speedup = scipy_ms / kernel_ms if kernel_ms > 0 else 0.0

        print(f"{sec:<12}{total_samples:<12}{scipy_ms:<14.2f}{kernel_ms:<18.2f}{total_ms:<16.2f}{speedup:<16.2f}x")


# ---------------------------------------------------------------------------
# Főprogram futtatása
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not cuda.is_available():
        print("Hiba: Nem található elérhető CUDA GPU.")
        exit(1)

    device = cuda.get_current_device()
    print(f"Aktív GPU: {device.name.decode('utf-8') if isinstance(device.name, bytes) else device.name}\n")

    # Teszt jel: 2 másodperces többkomponensű szinuszjel + zaj
    sr = 44100
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False, dtype=np.float32)
    test_signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 1200 * t)

    # 1. Verifikáció
    verify_correctness(test_signal, window_name='hann', frame_len=512, hop_size=256, n_fft=512)

    # 2. Skálázási benchmark
    benchmark_stft(sample_rates_seconds=[2, 5, 10], sample_rate=sr, n_fft=512, hop_size=256)