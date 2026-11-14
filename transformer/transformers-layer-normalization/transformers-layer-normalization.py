import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """
    # Ensure float64 precision for all inputs
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    
    # Compute mean and population variance (ddof=0) along the final axis
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, ddof=0, keepdims=True)
    
    # Normalize and apply learned affine transformation
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
