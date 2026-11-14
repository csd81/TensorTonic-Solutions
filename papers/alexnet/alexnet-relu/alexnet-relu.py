import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    Returns the elementwise float64 ReLU output.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    return np.maximum(0.0, x_arr)
