import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_score_arr = np.asarray(y_score, dtype=float)
    
    # Compute element-wise hinge loss: max(0, margin - y_i * s_i)
    losses = np.maximum(0.0, margin - y_true_arr * y_score_arr)
    
    # Apply the requested reduction and cast to a standard Python float
    if reduction == "mean":
        return float(np.mean(losses))
    elif reduction == "sum":
        return float(np.sum(losses))
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")
