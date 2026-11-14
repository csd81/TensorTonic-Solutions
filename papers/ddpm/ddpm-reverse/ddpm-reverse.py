import numpy as np


def reverse_step(
    x_t: list,
    t: int,
    epsilon_pred: list,
    betas: list[float],
    z: list = None,
) -> list:
    """
    Returns x at timestep t - 1, rounded to four decimals.
    """
    betas_arr = np.asarray(betas, dtype=np.float64)
    alphas = 1.0 - betas_arr
    alpha_bars = np.cumprod(alphas)

    idx = t - 1
    beta_t = betas_arr[idx]
    alpha_t = alphas[idx]
    alpha_bar_t = alpha_bars[idx]

    x_t_arr = np.asarray(x_t, dtype=np.float64)
    eps_pred_arr = np.asarray(epsilon_pred, dtype=np.float64)

    mean = (1.0 / np.sqrt(alpha_t)) * (
        x_t_arr - (beta_t / np.sqrt(1.0 - alpha_bar_t)) * eps_pred_arr
    )

    if t > 1:
        z_arr = np.asarray(z, dtype=np.float64)
        x_prev = mean + np.sqrt(beta_t) * z_arr
    else:
        x_prev = mean

    return np.round(x_prev, 4).tolist()
