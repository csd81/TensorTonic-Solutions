import numpy as np

def linear_regression_closed_form(X: list, y: list) -> list:
    """
    Returns the optimal weight vector as a list.
    """
    # Convert inputs to numpy arrays
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    # Compute the normal equation: w = (X^T * X)^-1 * X^T * y
    X_T = X_arr.T
    w = np.linalg.inv(X_T @ X_arr) @ X_T @ y_arr
    
    # Return as a list of floats
    return w.tolist()
