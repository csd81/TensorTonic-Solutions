import numpy as np

def cosine_noise_schedule(num_steps: int, s: float, max_beta: float) -> dict:
    """
    Returns a dict with float64 arrays "betas" and "alpha_bars", each of shape (num_steps,).
    """
    u = np.arange(num_steps + 1, dtype=np.float64)
    angles = ((u / num_steps + s) / (1.0 + s)) * (np.pi / 2.0)
    f = np.cos(angles) ** 2

    raw_betas = 1.0 - (f[1:] / f[:-1])
    betas = np.minimum(raw_betas, max_beta).astype(np.float64)
    alpha_bars = np.cumprod(1.0 - betas, dtype=np.float64)

    return {
        "betas": betas,
        "alpha_bars": alpha_bars,
    }
