import numpy as np

def normalize(data: list) -> np.ndarray:
    """
    Returns a float64 matrix standardized independently by column.
    """
    # Convert input data to a NumPy array with float64 precision
    X = np.array(data, dtype=np.float64)

    mean = np.mean(X, axis = 0)

    std = np.std(X, axis = 0)
    
    return (X - mean) / std