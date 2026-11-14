import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    """
    Returns the coefficient of determination as a Python float.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Residual sum of squares (RSS)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Total sum of squares (TSS)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Edge case: all targets are equal (ss_tot == 0)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return float(1.0 - (ss_res / ss_tot))
