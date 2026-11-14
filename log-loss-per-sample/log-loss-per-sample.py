import math

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """
    losses = []
    for y, p in zip(y_true, y_pred):
        # Clip predicted probability to [eps, 1 - eps]
        p_hat = min(1.0 - eps, max(eps, p))
        
        # Compute binary log loss using natural logarithm
        loss = -(y * math.log(p_hat) + (1 - y) * math.log(1.0 - p_hat))
        losses.append(float(loss))
        
    return losses
