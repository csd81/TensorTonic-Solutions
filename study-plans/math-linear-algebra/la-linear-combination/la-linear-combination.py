import numpy as np

def linear_combination(vectors: list, coefficients: list) -> np.ndarray:
    """
    Returns the weighted sum as a float64 vector.
    """
    # Convert inputs to NumPy arrays with float64 precision
    V = np.array(vectors, dtype=np.float64)
    c = np.array(coefficients, dtype=np.float64)
    
    # Perform a matrix-vector multiplication (c * V) 
    # to get the linear combination of the row vectors
    return c @ V
