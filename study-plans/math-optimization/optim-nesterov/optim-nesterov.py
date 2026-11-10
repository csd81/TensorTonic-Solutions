import numpy as np

def nesterov_momentum(X: list, y: list, lr: float, beta: float, n_epochs: int) -> dict:
    """
    Returns classical and Nesterov MSE loss curves in a dictionary.
    """
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)
    
    n_samples, n_features = X_arr.shape
    
    # -------------------------------------------------------------------------
    # 1. Run Classical Momentum
    # -------------------------------------------------------------------------
    w_classical = np.zeros(n_features, dtype=float)
    v_classical = np.zeros(n_features, dtype=float)
    classical_losses = []
    
    for _ in range(n_epochs):
        # Record loss before updating
        preds = X_arr @ w_classical
        mse = float(np.mean((preds - y_arr) ** 2))
        classical_losses.append(mse)
        
        # Gradient evaluated at current weights
        error = preds - y_arr
        gradient = (2.0 / n_samples) * (X_arr.T @ error)
        
        # Update rule
        v_classical = beta * v_classical + gradient
        w_classical -= lr * v_classical
        
    # -------------------------------------------------------------------------
    # 2. Run Nesterov Momentum
    # -------------------------------------------------------------------------
    w_nesterov = np.zeros(n_features, dtype=float)
    v_nesterov = np.zeros(n_features, dtype=float)
    nesterov_losses = []
    
    for _ in range(n_epochs):
        # Record loss before updating (at current weights, NOT look-ahead)
        preds = X_arr @ w_nesterov
        mse = float(np.mean((preds - y_arr) ** 2))
        nesterov_losses.append(mse)
        
        # Look-ahead point formulation
        w_look = w_nesterov - lr * beta * v_nesterov
        
        # Gradient evaluated at look-ahead point
        preds_look = X_arr @ w_look
        error_look = preds_look - y_arr
        gradient_look = (2.0 / n_samples) * (X_arr.T @ error_look)
        
        # Update rule
        v_nesterov = beta * v_nesterov + gradient_look
        w_nesterov -= lr * v_nesterov
        
    return {
        "classical_losses": classical_losses,
        "nesterov_losses": nesterov_losses
    }
