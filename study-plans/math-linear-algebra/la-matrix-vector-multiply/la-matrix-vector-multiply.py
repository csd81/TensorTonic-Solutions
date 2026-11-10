import numpy as np

def matrix_vector_multiply(A: list, x: list) -> np.ndarray:
    """
    Returns the matrix-vector product as a float64 array.
    """
    arr_A = np.array(A, dtype=np.float64)
    arr_x = np.array(x, dtype=np.float64)
    
    return arr_A @ arr_x
