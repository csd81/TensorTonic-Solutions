import numpy as np

def discriminator(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Returns discriminator probabilities as a float64 array with shape (B, 1).
    """
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    logits = x @ W
    probs = 1.0 / (1.0 + np.exp(-logits))
    
    return probs.astype(np.float64)
