import numpy as np

def adam(X: list, y: list, lr: float, beta1: float, beta2: float, n_epochs: int) -> dict:
    """
    Returns pre-update MSE losses and final weights in a dictionary.
    """
    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    
    n_samples, n_features = X_arr.shape
    
    # Initialize weights and moment vectors to zeros
    w = np.zeros(n_features, dtype=np.float64)
    m = np.zeros(n_features, dtype=np.float64)
    v = np.zeros(n_features, dtype=np.float64)
    
    losses = []
    eps = 1e-8
    
    for epoch in range(n_epochs):
        # Timestep t is 1-based index
        t = epoch + 1
        
        # 1. Record full-dataset MSE before the update
        preds = X_arr @ w
        residual = preds - y_arr
        losses.append(float(np.mean(residual ** 2)))
        
        # 2. Compute full-batch gradient
        gradient = (2.0 / n_samples) * (X_arr.T @ residual)
        
        # 3. Update biased first and second moment estimates
        m = beta1 * m + (1.0 - beta1) * gradient
        v = beta2 * v + (1.0 - beta2) * (gradient ** 2)
        
        # 4. Compute bias-corrected moment estimates
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        
        # 5. Update weights (eps goes OUTSIDE the sqrt for this specific problem)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
    return {
        "losses": losses,
        "final_weights": w.tolist()
    }
