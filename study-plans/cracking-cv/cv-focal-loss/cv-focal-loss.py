import numpy as np

def sigmoid_focal_loss(logits: list, targets: list, alpha: float, gamma: float) -> float:
    """
    Returns the mean focal loss as a float, rounded to 4 decimals.
    """
    # Convert inputs to high-precision NumPy arrays
    logits = np.array(logits, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    
    # 1. Compute numerically stable binary cross entropy directly from logits
    bce = np.maximum(logits, 0) - logits * targets + np.log(1.0 + np.exp(-np.abs(logits)))
    
    # 2. Compute probabilities for the modulating factor
    # Uses 1 / (1 + exp(-x)) for x >= 0 and exp(x) / (1 + exp(x)) for x < 0
    p = np.where(logits >= 0, 1.0 / (1.0 + np.exp(-logits)), np.exp(logits) / (1.0 + np.exp(logits)))
    
    # 3. Calculate p_t (the probability of the true target class)
    p_t = np.where(targets == 1, p, 1.0 - p)
    
    # 4. Calculate alpha_t balancing term (only if alpha >= 0 as per specification)
    if alpha >= 0:
        alpha_t = np.where(targets == 1, alpha, 1.0 - alpha)
        focal_loss = alpha_t * ((1.0 - p_t) ** gamma) * bce
    else:
        focal_loss = ((1.0 - p_t) ** gamma) * bce
        
    # 5. Get the mean loss over all elements and round to 4 decimal places
    mean_loss = np.mean(focal_loss)
    return float(round(mean_loss, 4))
