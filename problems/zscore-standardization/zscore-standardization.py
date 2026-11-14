import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    arr = np.asarray(X, dtype=float)
    
    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, ddof=0, keepdims=True)
    
    # Prevent division by zero where std <= eps, then zero out those slices
    safe_std = np.where(std <= eps, 1.0, std)
    z = (arr - mean) / safe_std
    
    return np.where(std <= eps, 0.0, z)
