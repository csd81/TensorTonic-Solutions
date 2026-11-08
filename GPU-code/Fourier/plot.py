import matplotlib
matplotlib.use("Agg")  # headless környezetben is működő mentéshez
import matplotlib.pyplot as plt
import numpy as np

from STFT import run_numba_stft


def plot_stft_comparison(x, sr, window_name='hann', frame_len=512, hop_size=256, n_fft=512):
    """
    Kiszámítja a SciPy és Numba STFT-t, majd vizuálisan összehasonlítja őket
    egy 3 paneles ábrán (SciPy, Numba GPU, Abszolút hiba).
    """
    from scipy import signal as sp_signal

    # Ablak és adatok előkészítése
    win = sp_signal.get_window(window_name, frame_len, fftbins=True).astype(np.float32)
    
    # 1. SciPy STFT referencia
    f_scipy, t_scipy, Zxx_scipy = sp_signal.stft(
        x, fs=sr, window=win, nperseg=frame_len,
        noverlap=frame_len - hop_size, nfft=n_fft,
        boundary=None, padded=False
    )
    scipy_mag_db = 20 * np.log10(np.abs(Zxx_scipy) + 1e-6)

    # 2. Numba GPU STFT (a korábbi függvény hívása)
    gpu_stft_result, _ = run_numba_stft(x, win, hop_size, n_fft)
    # A Numba kimenet alakja: [frekvencia_bin, idő_keret]
    gpu_mag_db = 20 * np.log10(np.abs(gpu_stft_result) + 1e-6)

    # Dimenziók illesztése az összehasonlításhoz
    min_frames = min(gpu_mag_db.shape[1], scipy_mag_db.shape[1])
    scipy_mag_db = scipy_mag_db[:, :min_frames]
    gpu_mag_db = gpu_mag_db[:, :min_frames]

    # Abszolút hiba a lineáris amplitúdókon vagy dB-ben
    abs_diff = np.abs(np.abs(gpu_stft_result[:, :min_frames]) - np.abs(Zxx_scipy[:, :min_frames]))

    # 3. Vizualizáció (Matplotlib)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    
    # Idő és frekvencia tengelyek határai
    extent = [t_scipy[0], t_scipy[min_frames-1], 0, sr / 2]

    # --- 1. Panel: SciPy CPU Referencia ---
    im0 = axes[0].imshow(scipy_mag_db, aspect='auto', origin='lower', extent=extent, cmap='inferno')
    axes[0].set_title("1. SciPy CPU Spektrogram")
    axes[0].set_xlabel("Idő [s]")
    axes[0].set_ylabel("Frekvencia [Hz]")
    fig.colorbar(im0, ax=axes[0], format='%+2.0f dB')

    # --- 2. Panel: Numba GPU Változat ---
    im1 = axes[1].imshow(gpu_mag_db, aspect='auto', origin='lower', extent=extent, cmap='inferno')
    axes[1].set_title("2. Numba GPU Spektrogram")
    axes[1].set_xlabel("Idő [s]")
    fig.colorbar(im1, ax=axes[1], format='%+2.0f dB')

    # --- 3. Panel: Abszolút Hibatérkép ---
    im2 = axes[2].imshow(abs_diff, aspect='auto', origin='lower', extent=extent, cmap='viridis')
    axes[2].set_title("3. Abszolút Eltérés (GPU vs. SciPy)")
    axes[2].set_xlabel("Idő [s]")
    fig.colorbar(im2, ax=axes[2], format='%.1e')

    plt.tight_layout()
    
    # Opcionális mentés a dokumentáció számára (pl. PDF vagy nagy felbontású PNG)
    plt.savefig("stft_gpu_vs_cpu_comparison.png", dpi=300, bbox_inches='tight')
    print("Az ábra sikeresen mentve: stft_gpu_vs_cpu_comparison.png")
    plt.close(fig)

# Futtatás példa:
if __name__ == "__main__":
    sr = 44100
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False, dtype=np.float32)
    test_signal = (
        np.sin(2 * np.pi * 440 * t)
        + 0.5 * np.sin(2 * np.pi * 1200 * t)
    ).astype(np.float32)
    plot_stft_comparison(test_signal, sr=sr)