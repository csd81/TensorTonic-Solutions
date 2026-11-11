import numpy as np

def outer_sum(a: list, b: list) -> np.ndarray:
    """
    Returns an (m, n) float64 array of pairwise sums.
    """
    # Convert lists to NumPy arrays with float64 dtype
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    
    # Reshape arr_a to a column vector (m, 1) and arr_b to a row vector (1, n)
    # The addition will trigger broadcasting to output an (m, n) matrix
    return arr_a[:, np.newaxis] + arr_b[np.newaxis, :]
