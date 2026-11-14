import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    """
    Returns the float64 latent sample with the same shape as mu.
    """
    std = np.exp(0.5 * log_var)
    z = mu + std * epsilon
    return np.asarray(z, dtype=np.float64)
