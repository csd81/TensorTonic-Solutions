import numpy as np

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    
    abs_error = np.abs(y_true_arr - y_pred_arr)
    
    quadratic = 0.5 * (abs_error ** 2)
    linear = delta * (abs_error - 0.5 * delta)
    
    loss = np.where(abs_error <= delta, quadratic, linear)
    return float(np.mean(loss))
