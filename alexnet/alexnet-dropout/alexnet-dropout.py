import numpy as np

def dropout(x: np.ndarray, p: float, training: bool,
            mask: np.ndarray) -> np.ndarray:
    """
    Returns the float64 inverted-dropout output.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    
    if not training:
        return x_arr.copy()
        
    mask_arr = np.asarray(mask, dtype=np.float64)
    return (x_arr * mask_arr) / (1.0 - p)
