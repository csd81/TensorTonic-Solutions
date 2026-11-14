import numpy as np

def dice_loss(p: list, y: list, eps: float = 1e-8) -> float:
    """
    Returns the loss as a float.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    
    intersection = np.sum(p_arr * y_arr)
    denominator = np.sum(p_arr) + np.sum(y_arr)
    
    dice_coeff = (2.0 * intersection + eps) / (denominator + eps)
    return float(1.0 - dice_coeff)
