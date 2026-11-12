import numpy as np

def vector_norms(v: list) -> np.ndarray:
    """
    Returns a float64 array containing the L1, L2, and infinity norms.
    """
    # Convert the input list to a NumPy array
    arr = np.array(v, dtype=np.float64)
    
    # Calculate each norm using np.linalg.norm
    l1 = np.linalg.norm(arr, ord=1)
    l2 = np.linalg.norm(arr, ord=2)
    linf = np.linalg.norm(arr, ord=np.inf)
    
    # Return as a NumPy float64 array of shape (3,)
    return np.array([l1, l2, linf], dtype=np.float64)
