import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    T = np.asarray(T, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    
    # Track if the input is a single 1D point so we can restore shape (3,) later
    is_single = (pts.ndim == 1)
    if is_single:
        pts = pts[np.newaxis, :]  # Reshape (3,) -> (1, 3)
        
    # Append a column of ones to form homogeneous coordinates of shape (N, 4)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])
    
    # Apply transformation: (N, 4) @ (4, 4).T -> (N, 4)
    transformed_h = pts_h @ T.T
    
    # Extract the first three coordinates (x', y', z')
    result = transformed_h[:, :3]
    
    return result[0] if is_single else result
