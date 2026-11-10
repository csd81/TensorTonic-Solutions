import numpy as np

def lasso_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
    """
    Fits linear regression with batch gradient descent and L1 regularization (Lasso).
    Returns the fitted weight list and bias rounded to four decimal places.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n, d = X.shape
    
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    
    for _ in range(epochs):
        y_pred = np.dot(X, w) + b
        error = y_pred - y
        
        # Gradient of MSE is 2/n * X^T * error
        # L1 subgradient adds alpha * sign(w), with sign(0) = 0
        dw = (2.0 / n) * np.dot(X.T, error) + alpha * np.sign(w)
        db = (2.0 / n) * np.sum(error)
        
        w -= lr * dw
        b -= lr * db
        
    return np.round(w, 4).tolist(), round(b, 4)
