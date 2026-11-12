import numpy as np

def rescale_zero_terminal_snr(betas: np.ndarray) -> dict:
    """
    Returns a dict with float64 arrays "betas", "alpha_bars" and "snr" of the rescaled schedule, each of shape (T,).
    """
    betas = np.asarray(betas, dtype=np.float64)
    
    # 1. Compute original cumulative signal fractions (alpha_bars) and their square roots
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    a = np.sqrt(alpha_bars)
    
    # 2. Shift and scale 'a' so that a_tilde[-1] = 0 and a_tilde[0] = a[0]
    a_0 = a[0]
    a_T = a[-1]
    a_tilde = (a - a_T) * (a_0 / (a_0 - a_T))
    
    # 3. Square to get the rescaled alpha_bars
    alpha_bars_tilde = a_tilde ** 2
    
    # 4. Recover new betas using shifted alpha_bars (with alpha_bar_{-1} = 1.0)
    alpha_bars_prev = np.concatenate(([1.0], alpha_bars_tilde[:-1]))
    betas_tilde = 1.0 - (alpha_bars_tilde / alpha_bars_prev)
    
    # 5. Compute the new signal-to-noise ratio (SNR)
    snr = alpha_bars_tilde / (1.0 - alpha_bars_tilde)
    
    return {
        "betas": betas_tilde,
        "alpha_bars": alpha_bars_tilde,
        "snr": snr
    }
