import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Computes the Mean Squared Error (MSE) between predictions and targets.
    
    Parameters:
    y_pred (list or np.ndarray): Predicted values.
    y_true (list or np.ndarray): Target (ground truth) values.
    
    Returns:
    float: The calculated MSE.
    """
    # Convert inputs to numpy arrays to handle vector operations
    predictions = np.asarray(y_pred)
    targets = np.asarray(y_true)
    
    # Calculate the average of the squared differences
    mse = np.mean((predictions - targets) ** 2)
    
    return float(mse)
