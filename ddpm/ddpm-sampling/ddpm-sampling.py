import numpy as np


def ddpm_sample(
    x_T: list, betas: list[float], epsilon_preds: list, z_values: list
) -> list:
    """
    Returns the final denoised sample rounded to four decimals.
    """
    betas_arr = np.asarray(betas, dtype=np.float64)
    alphas = 1.0 - betas_arr
    alpha_bars = np.cumprod(alphas)

    x_t = np.asarray(x_T, dtype=np.float64).copy()
    eps_arr = np.asarray(epsilon_preds, dtype=np.float64)
    z_arr = np.asarray(z_values, dtype=np.float64)

    T = len(betas_arr)

    for i in range(T):
        t_idx = T - 1 - i  # Zero-based schedule index for timestep t = T - i
        beta_t = betas_arr[t_idx]
        alpha_t = alphas[t_idx]
        alpha_bar_t = alpha_bars[t_idx]

        mean = (1.0 / np.sqrt(alpha_t)) * (
            x_t - (beta_t / np.sqrt(1.0 - alpha_bar_t)) * eps_arr[i]
        )

        if t_idx > 0:  # Equivalent to t > 1
            x_t = mean + np.sqrt(beta_t) * z_arr[i]
        else:
            x_t = mean

    return np.round(x_t, 4).tolist()
