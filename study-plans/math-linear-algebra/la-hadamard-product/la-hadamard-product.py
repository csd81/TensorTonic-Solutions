import numpy as np

def hadamard_product(A: list, B: list) -> np.ndarray:
    """
    Returns the element-wise product as a float64 array.
    """
    arr_A = np.array(A, dtype=np.float64)
    arr_B = np.array(B, dtype=np.float64)
    return arr_A * arr_B