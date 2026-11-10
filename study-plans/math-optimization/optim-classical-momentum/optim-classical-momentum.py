import numpy as np

def momentum_gd(X: list, y: list, lr: float, beta: float, n_epochs: int) -> dict:
    """
    Returns vanilla and momentum MSE loss curves in a dictionary.
    """
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)
    
    n_samples, n_features = X_arr.shape
    
    # -------------------------------------------------------------------------
    # 1. Run Vanilla Gradient Descent
    # -------------------------------------------------------------------------
    w_vanilla = np.zeros(n_features, dtype=float)
    vanilla_losses = []
    
    for _ in range(n_epochs):
        # Record loss BEFORE the update
        preds = X_arr @ w_vanilla
        mse = float(np.mean((preds - y_arr) ** 2))
        vanilla_losses.append(mse)
        
        # Compute full-batch gradient
        error = preds - y_arr
        gradient = (2.0 / n_samples) * (X_arr.T @ error)
        
        # Update weights
        w_vanilla -= lr * gradient
        
    # -------------------------------------------------------------------------
    # 2. Run Classical Momentum
    # -------------------------------------------------------------------------
    w_momentum = np.zeros(n_features, dtype=float)
    velocity = np.zeros(n_features, dtype=float)
    momentum_losses = []
    
    for _ in range(n_epochs):
        # Record loss BEFORE the update
        preds = X_arr @ w_momentum
        mse = float(np.mean((preds - y_arr) ** 2))
        momentum_losses.append(mse)
        
        # Compute full-batch gradient
        error = preds - y_arr
        gradient = (2.0 / n_samples) * (X_arr.T @ error)
        
        # Update velocity and weights
        velocity = beta * velocity + gradient
        w_momentum -= lr * velocity
        
    return {
        "vanilla_losses": vanilla_losses,
        "momentum_losses": momentum_losses
    }
