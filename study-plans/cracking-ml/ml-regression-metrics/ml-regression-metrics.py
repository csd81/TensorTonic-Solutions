import numpy as np

def regression_metrics(y_true: list, y_pred: list) -> dict:
    """
    Returns MSE, MAE, and R-squared in a dictionary.
    """
    # Convert inputs to NumPy float64 arrays
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    # Compute the residual vector (y_true - y_pred)
    residuals = y_true - y_pred
    
    # Compute MSE and MAE
    mse = np.mean(residuals ** 2)
    mae = np.mean(np.abs(residuals))
    
    # Compute Total Sum of Squares (TSS) and Residual Sum of Squares (RSS)
    rss = np.sum(residuals ** 2)
    y_bar = np.mean(y_true)
    tss = np.sum((y_true - y_bar) ** 2)
    
    # Handle zero total sum of squares explicitly
    if tss == 0:
        r2 = 0.0
    else:
        r2 = 1.0 - (rss / tss)
        
    # Return dictionary with rounded Python floats
    return {
        "mse": round(float(mse), 4),
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4)
    }
