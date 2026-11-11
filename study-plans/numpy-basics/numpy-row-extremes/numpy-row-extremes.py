import numpy as np

def row_extremes(data: list) -> np.ndarray:
    """
    Returns float64 rows of maxima, maximum indices, minima, and minimum indices.
    """
    # Convert input list to a NumPy array
    arr = np.array(data)
    
    # Calculate extremes and their indices along axis 1 (rows)
    # np.argmax and np.argmin automatically return the first occurrence (smallest index) on ties
    max_vals = np.max(arr, axis=1)
    max_idxs = np.argmax(arr, axis=1)
    min_vals = np.min(arr, axis=1)
    min_idxs = np.argmin(arr, axis=1)
    
    # Stack the results into a single array of shape (4, m) and cast to float64
    result = np.vstack([max_vals, max_idxs, min_vals, min_idxs]).astype(np.float64)
    
    return result
