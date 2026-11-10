import numpy as np

def svm_hinge_sgd(X: list, y: list, lr: float, lam: float, n_epochs: int) -> dict:
    """
    Trains a linear SVM using sequential Stochastic Gradient Descent with Hinge Loss.
    Returns weights, bias (rounded to 4 decimal places), and training set predictions.
    """
    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    n_samples, n_features = X_arr.shape
    
    # Initialize weights and bias to zero
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0
    
    # Sequential SGD training without shuffling
    for epoch in range(n_epochs):
        for i in range(n_samples):
            x_i = X_arr[i]
            y_i = y_arr[i]
            
            # Calculate signed margin
            margin = y_i * (np.dot(x_i, w) + b)
            
            if margin < 1.0:
                # Update rule for margin below one
                w = w - lr * (lam * w - y_i * x_i)
                b = b + lr * y_i
            else:
                # Update rule for margin at least one
                w = w - lr * lam * w
                
    # Final training set predictions: +1 for strictly positive score, -1 otherwise
    scores = np.dot(X_arr, w) + b
    predictions = [1 if s > 0 else -1 for s in scores]
    
    # Round parameters to four decimal places and convert weights to list
    return {
        "bias": float(round(b, 4)),
        "predictions": predictions,
        "weights": [float(round(val, 4)) for val in w.tolist()]
    }
