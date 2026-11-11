import numpy as np

def winsorize(data: list, lo_q: float, hi_q: float) -> np.ndarray:
    """
    Returns float64 slices of clipped values, lower mask, and upper mask.
    """
    # Convert input to a float64 NumPy array
    arr = np.array(data, dtype=np.float64)
    
    # Calculate bounds along each column (axis=0) independently
    # keepdims=True ensures proper broadcasting shape (1, n) against (m, n)
    lo_bounds = np.percentile(arr, lo_q, axis=0, keepdims=True)
    hi_bounds = np.percentile(arr, hi_q, axis=0, keepdims=True)
    
    # Compute the elements
    clipped = np.clip(arr, lo_bounds, hi_bounds)
    lo_mask = (arr < lo_bounds).astype(np.float64)
    hi_mask = (arr > hi_bounds).astype(np.float64)
    
    # Stack along a new outer axis to yield shape (3, m, n)
    return np.stack([clipped, lo_mask, hi_mask], axis=0)
