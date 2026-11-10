import numpy as np

def dice_loss(pred: list, target: list, smooth: float) -> float:
    """
    Returns the Dice loss as a float, rounded to 4 decimals.
    """
    # Convert inputs to float64 numpy arrays
    p = np.array(pred, dtype=np.float64)
    t = np.array(target, dtype=np.float64)
    
    # Calculate sums
    sum_p = np.sum(p)
    sum_t = np.sum(t)
    
    # Edge case: If both matrices contain only zeros, return 0.0
    if sum_p == 0.0 and sum_t == 0.0:
        return 0.0
        
    # Calculate intersection
    intersection = np.sum(p * t)
    
    # Calculate dice coefficient and loss
    dice_coef = (2.0 * intersection + smooth) / (sum_p + sum_t + smooth)
    loss = 1.0 - dice_coef
    
    return round(float(loss), 4)
