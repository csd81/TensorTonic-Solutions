import numpy as np

def vae_decoder(z: np.ndarray, W_dec: np.ndarray, b_dec: np.ndarray) -> np.ndarray:
    """
    Returns the float64 reconstruction with shape (B, D).
    """
    logits = z @ W_dec + b_dec
    x_hat = 1.0 / (1.0 + np.exp(-logits))
    return np.asarray(x_hat, dtype=np.float64)
