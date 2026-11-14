import numpy as np

def rotate_around_z(points: list, theta: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as points.
    """
    pts = np.asarray(points, dtype=float)
    
    c = np.cos(theta)
    s = np.sin(theta)
    
    x = pts[..., 0]
    y = pts[..., 1]
    z = pts[..., 2]
    
    x_new = x * c - y * s
    y_new = x * s + y * c
    
    return np.stack([x_new, y_new, z], axis=-1)
