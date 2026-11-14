import numpy as np

def vae_encoder(
    x: np.ndarray,
    W_mu: np.ndarray,
    b_mu: np.ndarray,
    W_logvar: np.ndarray,
    b_logvar: np.ndarray,
) -> dict:
    """
    Returns mu and log_var as float64 arrays in a dictionary.
    """
    mu = np.asarray(x @ W_mu + b_mu, dtype=np.float64)
    log_var = np.asarray(x @ W_logvar + b_logvar, dtype=np.float64)

    return {
        "mu": mu,
        "log_var": log_var,
    }
