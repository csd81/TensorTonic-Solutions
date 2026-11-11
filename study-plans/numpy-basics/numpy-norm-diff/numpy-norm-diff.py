import numpy as np

def norm_diff(a: list, b: list, lo: float, hi: float) -> np.ndarray:
    """
    Returns a float64 array of absolute normalized differences.
    """
    # Clip the input arrays within the given interval [lo, hi]
    clipped_a = np.clip(a, lo, hi)
    clipped_b = np.clip(b, lo, hi)
    
    # Calculate the absolute difference and rescale by the interval range
    D = np.abs(clipped_a - clipped_b) / (hi - lo)
    
    # Ensure the returned array is explicitly of type float64
    return D.astype(np.float64)
