import numpy as np

def scale_cols(data: list, weights: list) -> np.ndarray:
    """
    Returns a float64 matrix with each column multiplied by its weight.
    """
    # Convert inputs to NumPy arrays and explicitly set dtype to float64
    X = np.array(data, dtype=np.float64)
    w = np.array(weights, dtype=np.float64)
    
    # NumPy automatically broadcasts the 1D array 'w' across the rows of 'X'
    return X * w
