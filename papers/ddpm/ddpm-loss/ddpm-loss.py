import numpy as np


def compute_ddpm_loss(epsilon: list, epsilon_pred: list) -> float:
    """
    Returns the mean DDPM noise-prediction loss.
    """
    eps_arr = np.asarray(epsilon, dtype=np.float64)
    eps_pred_arr = np.asarray(epsilon_pred, dtype=np.float64)

    mse = np.mean((eps_arr - eps_pred_arr) ** 2)
    return round(float(mse), 6)
