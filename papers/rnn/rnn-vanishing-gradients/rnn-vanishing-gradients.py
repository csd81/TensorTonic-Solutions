import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> np.ndarray:
    """
    Returns T float64 spectral-norm powers.
    """
    # Compute the spectral norm (largest singular value / 2-norm) of W_hh
    rho = np.linalg.norm(W_hh, ord=2)
    
    # Generate exponents [0, 1, ..., T - 1]
    powers = np.arange(T, dtype=np.float64)
    
    # Raise rho to each power and ensure float64 dtype
    return np.power(rho, powers, dtype=np.float64)
