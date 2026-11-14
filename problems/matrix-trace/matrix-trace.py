import numpy as np


def matrix_trace(A: list) -> float:
    """
    Returns the trace as a float.
    """
    arr = np.asarray(A, dtype=float)
    idx = np.arange(arr.shape[0])
    return float(np.sum(arr[idx, idx]))
