import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    # Convert inputs to float64 NumPy arrays
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n, d = X.shape
    
    # Initialize weights and bias to zero
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    
    # Batch Gradient Descent loop
    for _ in range(epochs):
        # Generate predictions
        y_pred = np.dot(X, w) + b
        error = y_pred - y
        
        # Calculate gradients with the 2/n factor
        dw = (2 / n) * np.dot(X.T, error)
        db = (2 / n) * np.sum(error)
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
        
    # Round results to 4 decimal places and format as requested
    return np.round(w, 4).tolist(), round(b, 4)
