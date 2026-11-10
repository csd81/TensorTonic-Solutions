import numpy as np

def log_loss(y_true: list, y_pred: list) -> float:
    """
    Returns the mean binary log loss.
    """
    # Convert inputs to float64 NumPy arrays
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    # Clip probabilities to prevent log(0) and log(1) issues
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Compute the binary cross-entropy loss
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # Return as a Python float rounded to four decimal places
    return round(float(loss), 4)
