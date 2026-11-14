import numpy as np

def swish(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    
    # Compute sigmoid safely to avoid exponential overflow
    # For x >= 0: 1 / (1 + exp(-x))
    # For x < 0:  exp(x) / (1 + exp(x))
    pos_mask = x_arr >= 0
    sigmoid = np.empty_like(x_arr)
    
    sigmoid[pos_mask] = 1.0 / (1.0 + np.exp(-x_arr[pos_mask]))
    
    exp_x = np.exp(x_arr[~pos_mask])
    sigmoid[~pos_mask] = exp_x / (1.0 + exp_x)
    
    return x_arr * sigmoid
