import numpy as np

def tanh(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    return np.tanh(x_arr)
