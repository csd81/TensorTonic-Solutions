import numpy as np

def newtons_method(X: list, y: list, lr_gd: float, n_iters: int) -> dict:
    """
    Returns gradient-descent and Newton MSE curves in a dictionary.
    """
    # 1. Convert to float64 NumPy arrays
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    
    n = X.shape[0]
    
    # 2. Append a column of ones for the bias term
    X_aug = np.hstack([X, np.ones((n, 1), dtype=np.float64)])
    
    # Initialize parameters to zeros
    w_gd = np.zeros(X_aug.shape[1], dtype=np.float64)
    w_newton = np.zeros(X_aug.shape[1], dtype=np.float64)
    
    # Helper to calculate Mean Squared Error (MSE)
    def compute_mse(w):
        error = X_aug @ w - y
        return float(np.mean(error ** 2))
    
    # Record initial MSE losses
    gd_losses = [compute_mse(w_gd)]
    newton_losses = [compute_mse(w_newton)]
    
    # Precompute Hessian for Newton's method (constant for linear regression MSE)
    # H = (2 / n) * X_aug.T @ X_aug
    H = (2.0 / n) * (X_aug.T @ X_aug)
    
    # Run optimization loops
    for _ in range(n_iters):
        # --- Gradient Descent Update ---
        g_gd = (2.0 / n) * (X_aug.T @ (X_aug @ w_gd - y))
        w_gd -= lr_gd * g_gd
        gd_losses.append(compute_mse(w_gd))
        
        # --- Newton's Method Update ---
        g_newton = (2.0 / n) * (X_aug.T @ (X_aug @ w_newton - y))
        # Solve H * delta = g instead of inverting H
        delta = np.linalg.solve(H, g_newton)
        w_newton -= delta
        newton_losses.append(compute_mse(w_newton))
        
    return {
        "gd_losses": gd_losses,
        "newton_losses": newton_losses
    }
