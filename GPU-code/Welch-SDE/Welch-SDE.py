"""Welch power spectral density (PSD) estimation with Numba CUDA.

Two kernels:
  1. ``welch_fft_kernel``   - windows each segment and computes its FFT with a
                              radix-2 Cooley-Tukey kernel in shared memory.
  2. ``welch_psd_kernel``   - averages |X_i[k]|^2 over segments and applies the
                              Welch density scaling.

Only NumPy and Numba are needed at runtime. The demo uses SciPy, when
available, as the numerical reference.

Usage:
    freqs, psd = welch_psd(signal, fs=1000.0, nperseg=256,
                           noverlap=128, window="hann", onesided=True)

Constraints: nperseg must be a power of two with 2 <= nperseg <= 1024
(one block per segment, no cross-block FFT).
"""

import numpy as np
from numba import cuda, float32, int32

THREADS_PSD = 256


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@cuda.jit
def welch_fft_kernel(signal, window, bitrev, tw_cos, tw_sin,
                     out_re, out_im, n_seg, L, D):
    """Window one segment and run a radix-2 DIT FFT, one block per segment.

    Dynamic shared memory holds interleaved re/im planes: re in [0, L),
    im in [L, 2L).
    """
    seg = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    smem = cuda.shared.array(0, float32)

    if seg < n_seg and tid < L:
        smem[tid] = signal[seg * D + tid] * window[tid]
        smem[L + tid] = 0.0
    cuda.syncthreads()

    # Bit-reversal permutation
    if tid < L:
        j = bitrev[tid]
        if j > tid:
            tr = smem[tid]
            ti = smem[L + tid]
            smem[tid] = smem[j]
            smem[L + tid] = smem[L + j]
            smem[j] = tr
            smem[L + j] = ti
    cuda.syncthreads()

    # Butterfly stages: length = 2, 4, ..., L
    length = 2
    while length <= L:
        half = length // 2
        if tid < L // 2:
            group = tid // half
            offset = tid % half
            i0 = group * length + offset
            i1 = i0 + half
            m = offset * (L // length)
            wr = tw_cos[m]
            wi = tw_sin[m]

            br = smem[i1]
            bi = smem[L + i1]
            tr = wr * br - wi * bi
            ti = wr * bi + wi * br
            smem[L + i1] = smem[L + i0] - ti
            smem[i1] = smem[i0] - tr
            smem[L + i0] += ti
            smem[i0] += tr
        cuda.syncthreads()
        length *= 2

    if tid < L:
        out_re[seg, tid] = smem[tid]
        out_im[seg, tid] = smem[L + tid]


@cuda.jit
def welch_psd_kernel(re, im, out, n_seg, n_freq, scale, onesided,
                     nyquist_idx):
    """Average |X_i[k]|^2 over segments for every frequency bin k."""
    k = cuda.grid(1)
    if k < n_freq:
        total = 0.0
        for i in range(n_seg):
            a = re[i, k]
            b = im[i, k]
            total += a * a + b * b

        factor = scale
        if onesided and k > 0 and k != nyquist_idx:
            factor *= 2.0
        out[k] = total * factor


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------

def _make_window(name, L):
    n = np.arange(L, dtype=np.float64)
    if name == "hann":
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / L)
    elif name == "hamming":
        w = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / L)
    elif name in ("rect", "boxcar"):
        w = np.ones(L, dtype=np.float64)
    else:
        raise ValueError(f"unknown window {name!r}; use 'hann', 'hamming' or 'rect'")
    return w.astype(np.float32)


def _bit_reverse_table(L):
    bits = L.bit_length() - 1
    rev = np.empty(L, dtype=np.int32)
    for i in range(L):
        value = i
        result = 0
        for _ in range(bits):
            result = (result << 1) | (value & 1)
            value >>= 1
        rev[i] = result
    return rev


def _twiddle_tables(L):
    m = np.arange(L // 2, dtype=np.float64)
    angle = 2.0 * np.pi * m / L
    return np.cos(angle).astype(np.float32), (-np.sin(angle)).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def welch_psd(signal, fs=1.0, nperseg=256, noverlap=None, window="hann",
              onesided=True):
    """Welch PSD estimate.

    Returns (frequencies, psd). ``psd`` is one-sided (length nperseg/2 + 1)
    by default, matching scipy.signal.welch(..., scaling="density",
    detrend=False); set ``onesided=False`` for the full two-sided spectrum.
    """
    x = np.ascontiguousarray(signal, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {x.shape}")

    N = x.size
    L = int(nperseg)
    if not 2 <= L <= 1024:
        raise ValueError(f"nperseg must satisfy 2 <= nperseg <= 1024, got {L}")
    if L & (L - 1):
        raise ValueError(f"nperseg must be a power of two, got {L}")

    if noverlap is None:
        noverlap = L // 2
    noverlap = int(noverlap)
    if not 0 <= noverlap < L:
        raise ValueError(f"noverlap must satisfy 0 <= noverlap < nperseg, got {noverlap}")

    step = L - noverlap
    if N < L:
        raise ValueError(f"signal length {N} is shorter than nperseg {L}")
    n_seg = 1 + (N - L) // step
    if n_seg < 1:
        raise ValueError("no complete segments in the signal")

    w = _make_window(window, L)
    sum_w2 = float(np.sum(w.astype(np.float64) ** 2))
    bitrev = _bit_reverse_table(L)
    tw_cos, tw_sin = _twiddle_tables(L)

    d_signal = cuda.to_device(x)
    d_window = cuda.to_device(w)
    d_bitrev = cuda.to_device(bitrev)
    d_tw_cos = cuda.to_device(tw_cos)
    d_tw_sin = cuda.to_device(tw_sin)
    d_re = cuda.device_array((n_seg, L), dtype=np.float32)
    d_im = cuda.device_array((n_seg, L), dtype=np.float32)

    # One block per segment, one thread per sample, dynamic re/im shared memory.
    welch_fft_kernel[n_seg, L, 0, 2 * L * 4](
        d_signal, d_window, d_bitrev, d_tw_cos, d_tw_sin,
        d_re, d_im, n_seg, L, step,
    )

    if onesided:
        n_freq = L // 2 + 1
        nyquist_idx = L // 2 if L % 2 == 0 else -1
    else:
        n_freq = L
        nyquist_idx = -1

    d_psd = cuda.device_array(n_freq, dtype=np.float32)
    blocks = (n_freq + THREADS_PSD - 1) // THREADS_PSD
    scale = 1.0 / (n_seg * fs * sum_w2)
    welch_psd_kernel[blocks, THREADS_PSD](
        d_re, d_im, d_psd, n_seg, n_freq, scale, int(onesided), nyquist_idx,
    )
    cuda.synchronize()

    psd = d_psd.copy_to_host()
    if onesided:
        freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    else:
        freqs = np.fft.fftfreq(L, d=1.0 / fs)
    return freqs, psd


# ---------------------------------------------------------------------------
# Demo / validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    fs = 1000.0
    N = 1 << 16
    L = 256
    noverlap = L // 2

    t = np.arange(N) / fs
    rng = np.random.default_rng(0)
    x = (
        0.7 * np.sin(2.0 * np.pi * 50.0 * t)
        + 0.5 * np.sin(2.0 * np.pi * 120.0 * t)
        + 0.1 * rng.standard_normal(N)
    ).astype(np.float32)

    # Warm-up: JIT compilation and allocator caches, excluded from timing.
    welch_psd(x, fs=fs, nperseg=L, noverlap=noverlap, window="hann")

    t0 = time.perf_counter()
    freqs, psd = welch_psd(x, fs=fs, nperseg=L, noverlap=noverlap, window="hann")
    gpu_s = time.perf_counter() - t0

    n_seg = 1 + (N - L) // (L - noverlap)
    print(f"N={N}, nperseg={L}, noverlap={noverlap}, segments={n_seg}")
    print(f"GPU Welch time (incl. transfers): {gpu_s * 1000:.2f} ms")
    print(f"peak frequency: {freqs[np.argmax(psd)]:.2f} Hz")

    try:
        from scipy.signal import welch

        t0 = time.perf_counter()
        f_ref, p_ref = welch(
            x, fs=fs, window="hann", nperseg=L, noverlap=noverlap,
            detrend=False, scaling="density", return_onesided=True,
        )
        cpu_s = time.perf_counter() - t0

        max_abs = float(np.max(np.abs(psd - p_ref)))
        max_rel = max_abs / float(np.max(np.abs(p_ref)))
        print(f"SciPy time: {cpu_s * 1000:.2f} ms")
        print(f"max abs error: {max_abs:.3e} | max rel error: {max_rel:.3e}")
    except ImportError:
        print("SciPy not installed; skipping reference comparison")
