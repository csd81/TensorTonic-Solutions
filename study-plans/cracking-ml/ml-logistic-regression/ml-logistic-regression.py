import numpy as np

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    """
    Fits binary logistic regression with batch gradient descent.
    Returns the fitted weight list and bias rounded to four decimal places.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n, d = X.shape
    
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    
    for _ in range(n_iters):
        z = np.dot(X, w) + b
        z = np.clip(z, -500.0, 500.0)
        p = 1.0 / (1.0 + np.exp(-z))
        
        dw = np.dot(X.T, (p - y)) / n
        db = np.mean(p - y)
        
        w -= lr * dw
        b -= lr * db
        
    return np.round(w, 4).tolist(), round(b, 4)
