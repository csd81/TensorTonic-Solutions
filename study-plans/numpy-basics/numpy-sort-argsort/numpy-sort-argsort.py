import numpy as np

def sort_with_indices(data: list, axis: int) -> np.ndarray:
    """
    Returns a (2, m, n) float64 array of sorted values and source indices.
    """
    # Convert input to a numpy array
    arr = np.array(data, dtype=np.float64)
    
    # Get the sorted values and original indices along the specified axis
    sorted_vals = np.sort(arr, axis=axis)
    argsort_indices = np.argsort(arr, axis=axis).astype(np.float64)
    
    # Stack them along a new first axis to achieve the (2, m, n) shape
    return np.stack([sorted_vals, argsort_indices], axis=0)
