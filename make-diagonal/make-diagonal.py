import numpy as np

def make_diagonal(v: list) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, N).
    """
    arr = np.asarray(v)
    n = arr.shape[0]
    D = np.zeros((n, n), dtype=arr.dtype)
    idx = np.arange(n)
    D[idx, idx] = arr
    return D
