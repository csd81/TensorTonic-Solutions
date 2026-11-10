import numpy as np

def percentiles(x: list, q: list) -> np.ndarray:
    """
    Returns the requested percentiles as a float64 array.
    """
    x_sorted = sorted(x)
    n = len(x)
    results = []
    for p in q:
        h = (p/100.0) * (n-1)
        idx_lower = int(np.floor(h))
        idx_upper = int(np.ceil(h))
        if idx_lower == idx_upper:
            val = float(x_sorted[idx_lower])
        else:
            fraction = h-idx_lower
            val = x_sorted[idx_lower] + fraction * (x_sorted[idx_upper]-x_sorted[idx_lower])
        
        results.append(val)
    return np.array(results, dtype=np.float64)
        