import numpy as np

def percentiles(x: list, q: list) -> np.ndarray:
    """
    Returns a NumPy array of percentiles.
    """
    x_sorted = np.sort(x)
    n = len(x_sorted)
    
    # Convert percentiles q to zero-based positions r
    r = (np.array(q, dtype=float) / 100.0) * (n - 1)
    
    l = np.floor(r).astype(int)
    u = np.ceil(r).astype(int)
    w = r - l
    
    # Interpolate between sorted values
    p_q = (1.0 - w) * x_sorted[l] + w * x_sorted[u]
    
    return np.array(p_q)
