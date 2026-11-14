import numpy as np

def angle_between_3d(v: list, w: list) -> float:
    """
    Returns the angle as a float.
    """
    v_arr = np.asarray(v, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    
    norm_v = np.linalg.norm(v_arr)
    norm_w = np.linalg.norm(w_arr)
    
    if norm_v == 0.0 or norm_w == 0.0:
        return np.nan
        
    cos_theta = np.dot(v_arr, w_arr) / (norm_v * norm_w)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    return float(np.arccos(cos_theta))
