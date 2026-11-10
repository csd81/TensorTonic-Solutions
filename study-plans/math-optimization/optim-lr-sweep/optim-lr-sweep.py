import numpy as np

def learning_rate_sweep(X: list, y: list, learning_rates: list, n_epochs: int) -> list:
    """
    Returns one pre-update MSE loss curve per learning rate.
    """
    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    
    n_samples, n_features = X_arr.shape
    results = []
    
    for lr in learning_rates:
        # Restart weights at zero for every learning rate candidate
        w = np.zeros(n_features, dtype=np.float64)
        loss_curve = []
        
        for epoch in range(n_epochs):
            # 1. Record MSE before each update step
            preds = X_arr @ w
            residual = preds - y_arr
            loss_curve.append(float(np.mean(residual ** 2)))
            
            # 2. Compute full-batch data gradient
            gradient = (2.0 / n_samples) * (X_arr.T @ residual)
            
            # 3. Update weights vector
            w -= lr * gradient
            
        results.append(loss_curve)
        
    return results
