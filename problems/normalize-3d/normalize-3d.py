import numpy as np

def normalize_3d(v: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as v.
    """
    arr = np.asarray(v, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return np.where(norms > 0, arr / np.where(norms > 0, norms, 1.0), 0.0)
