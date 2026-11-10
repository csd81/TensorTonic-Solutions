import numpy as np

def rmsprop(X: list, y: list, lr: float, decay: float, n_epochs: int) -> dict:
    """
    Returns MSE losses and effective learning rates in a dictionary.
    """
    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    
    n_samples, n_features = X_arr.shape
    
    # Initialize weights and the squared gradient running average to zero vectors
    w = np.zeros(n_features, dtype=np.float64)
    E = np.zeros(n_features, dtype=np.float64)
    
    losses = []
    effective_lrs = []
    eps = 1e-8
    
    for epoch in range(n_epochs):
        # 1. Record full-dataset MSE before the update
        preds = X_arr @ w
        residual = preds - y_arr
        losses.append(float(np.mean(residual ** 2)))
        
        # 2. Compute full-batch gradient
        gradient = (2.0 / n_samples) * (X_arr.T @ residual)
        
        # 3. Update the running average of squared gradients (E_t)
        E = decay * E + (1.0 - decay) * (gradient ** 2)
        
        # 4. Record the effective learning rates (eps inside the sqrt)
        eff_lr = lr / np.sqrt(E + eps)
        effective_lrs.append(eff_lr.tolist())
        
        # 5. Update weights
        w -= eff_lr * gradient
        
    return {
        "losses": losses,
        "effective_lrs": effective_lrs
    }
