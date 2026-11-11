import numpy as np

def scale_rows(data: list, weights: list) -> np.ndarray:
    """
    Returns a float64 matrix with each row multiplied by its weight.
    """
    # Convert inputs to NumPy arrays, forcing float64 as required
    data_arr = np.array(data, dtype=np.float64)
    weights_arr = np.array(weights, dtype=np.float64)
    
    # Reshape weights from (N,) to (N, 1) to enable row-by-row broadcasting
    return data_arr * weights_arr[:, np.newaxis]
