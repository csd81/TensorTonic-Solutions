import numpy as np

def matrix_multiply(A: list, B: list) -> np.ndarray:
    """
    Returns the matrix product as a float64 array.
    """
    arr_A = np.array(A, dtype=np.float64)
    arr_B = np.array(B, dtype=np.float64)
    
    return arr_A @ arr_B
