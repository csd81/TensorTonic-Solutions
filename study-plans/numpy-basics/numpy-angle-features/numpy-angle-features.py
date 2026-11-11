import numpy as np

def angle_features(angles: list) -> np.ndarray:
    """
    Returns a (3, n) float64 array with sine, cosine, and tangent rows.
    """
    # Convert the input list to a NumPy array with float64 precision
    arr = np.array(angles, dtype=np.float64)
    
    # Compute sine, cosine, and tangent vertically stacked into a (3, n) array
    return np.vstack([np.sin(arr), np.cos(arr), np.tan(arr)])
