import numpy as np

def clip_gradients(g: list, max_norm: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as g.
    """
    g_arr = np.asarray(g, dtype=float)
    
    # Compute the global L2 norm across all elements regardless of shape
    global_norm = np.linalg.norm(g_arr)
    
    if global_norm > max_norm:
        scale = max_norm / global_norm
        return g_arr * scale
        
    return g_arr.copy()
