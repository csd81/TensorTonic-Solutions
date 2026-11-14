import numpy as np


def get_alpha_bar(betas: list[float]) -> list[float]:
    """Returns the cumulative alpha-bar values rounded to six decimals."""
    betas_arr = np.asarray(betas, dtype=np.float64)
    alphas = 1.0 - betas_arr
    alpha_bars = np.cumprod(alphas)
    return np.round(alpha_bars, 6).tolist()


def forward_diffusion(
    x_0: list, t: int, betas: list[float], epsilon: list
) -> list:
    """Returns x_t with the same nested shape as x_0."""
    betas_arr = np.asarray(betas, dtype=np.float64)
    alpha_bars = np.cumprod(1.0 - betas_arr)
    alpha_bar_t = alpha_bars[t - 1]

    x0_arr = np.asarray(x_0, dtype=np.float64)
    eps_arr = np.asarray(epsilon, dtype=np.float64)

    x_t = np.sqrt(alpha_bar_t) * x0_arr + np.sqrt(1.0 - alpha_bar_t) * eps_arr
    return np.round(x_t, 4).tolist()
