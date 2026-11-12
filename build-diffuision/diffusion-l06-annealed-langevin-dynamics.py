import numpy as np

def annealed_langevin_dynamics(x_init: np.ndarray, sigmas: np.ndarray, steps_per_sigma: int, step_size: float, noise: np.ndarray, data_mean: np.ndarray, data_var: float) -> np.ndarray:
    """
    Returns the final samples after annealed Langevin dynamics, a float64 array of shape (N, D).
    """
    x = np.array(x_init, dtype=np.float64, copy=True)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    data_mean = np.asarray(data_mean, dtype=np.float64)

    sigma_last_sq = sigmas[-1] ** 2

    for i, sigma in enumerate(sigmas):
        sigma_sq = sigma ** 2
        alpha = step_size * (sigma_sq / sigma_last_sq)
        drift_scale = 0.5 * alpha
        noise_scale = np.sqrt(alpha)
        denom = data_var + sigma_sq

        for j in range(steps_per_sigma):
            score = -(x - data_mean) / denom
            z = noise[i * steps_per_sigma + j]
            x = x + drift_scale * score + noise_scale * z

    return x
