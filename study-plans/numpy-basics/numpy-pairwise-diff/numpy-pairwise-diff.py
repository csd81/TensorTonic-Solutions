import numpy as np

def pairwise_diff(a: list) -> np.ndarray:
    """
    Returns an (n, n) float64 array of signed pairwise differences.
    """
    # Convert input to a numpy array with float64 data type
    arr = np.array(a, dtype=np.float64)
    
    # Use broadcasting: (n, 1) - (1, n) results in an (n, n) matrix
    return arr[:, np.newaxis] - arr[np.newaxis, :]
