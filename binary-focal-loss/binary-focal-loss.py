import math

def binary_focal_loss(predictions: list, targets: list, alpha: float, gamma: float) -> float:
    """
    Returns the mean binary focal loss as a float.
    """
    total_loss = 0.0
    n = len(predictions)
    
    for p, y in zip(predictions, targets):
        # Probability assigned to the true class
        p_t = p if y == 1 else (1.0 - p)
        
        # Focal loss formula: -alpha * (1 - p_t)^gamma * ln(p_t)
        loss = -alpha * ((1.0 - p_t) ** gamma) * math.log(p_t)
        total_loss += loss
        
    return float(total_loss / n)
