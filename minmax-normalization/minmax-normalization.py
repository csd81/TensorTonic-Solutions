import numpy as np

def minmax_scale(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns a floating-point NumPy array matching the shape of X.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    
    x_min = np.min(X_arr, axis=axis, keepdims=True)
    x_max = np.max(X_arr, axis=axis, keepdims=True)
    x_range = x_max - x_min
    
    # Avoid division by zero where range <= eps
    safe_range = np.where(x_range <= eps, 1.0, x_range)
    scaled = (X_arr - x_min) / safe_range
    
    # Set slices with range <= eps to 0.0
    return np.where(x_range <= eps, 0.0, scaled)
