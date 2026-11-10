import numpy as np

def matrix_determinant(A: list) -> float:
    """
    Returns the determinant as a Python float.
    """
    matrix =np.array(A, dtype=np.float64)

    det_value = np.linalg.det(matrix)

    return float(det_value)