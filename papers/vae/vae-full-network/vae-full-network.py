import numpy as np

def vae_forward(x: np.ndarray, epsilon: np.ndarray,
                W_mu: np.ndarray, b_mu: np.ndarray,
                W_logvar: np.ndarray, b_logvar: np.ndarray,
                W_dec: np.ndarray, b_dec: np.ndarray) -> dict:
    """
    Returns reconstruction, mu, log_var, and z as float64 arrays.
    """
    # 1. Encoder
    mu = np.asarray(x @ W_mu + b_mu, dtype=np.float64)
    log_var = np.asarray(x @ W_logvar + b_logvar, dtype=np.float64)

    # 2. Reparameterization Trick
    std = np.exp(0.5 * log_var)
    z = np.asarray(mu + std * epsilon, dtype=np.float64)

    # 3. Bernoulli Decoder
    logits = z @ W_dec + b_dec
    reconstruction = np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=np.float64)

    return {
        "reconstruction": reconstruction,
        "mu": mu,
        "log_var": log_var,
        "z": z,
    }
