import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    arr = np.asarray(A)
    n, m = arr.shape
    out = np.empty((m, n), dtype=arr.dtype)

    for j in range(m):
        out[j, :] = arr[:, j]

    return out
