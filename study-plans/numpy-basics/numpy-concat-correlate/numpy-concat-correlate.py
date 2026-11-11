import numpy as np

def compare_correlations(a: list, b: list) -> np.ndarray:
    """
    Returns float64 correlation matrices for a, b, and their combined rows.
    """
    # Convert inputs to float64 NumPy arrays
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    
    # Vertically stack them to get the combined dataset
    arr_combined = np.vstack((arr_a, arr_b))
    
    # np.corrcoef calculates correlation between rows by default.
    # Passing .T (transpose) forces it to calculate column-wise correlations.
    corr_a = np.corrcoef(arr_a.T)
    corr_b = np.corrcoef(arr_b.T)
    corr_combined = np.corrcoef(arr_combined.T)
    
    # Stack the three 2D matrices along a new 3D axis (shape: 3, n, n)
    return np.stack([corr_a, corr_b, corr_combined], axis=0)
