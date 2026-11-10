import numpy as np

def gradient_clipping(X: list, y: list, lr: float, clip_value: float, clip_norm: float, n_epochs: int) -> dict:
    """
    Returns unclipped, value-clipped, and norm-clipped MSE curves.
    """
    X = np.array(X)
    y = np.array(y)
    n, d = X.shape
    
    # --- 1. Unclipped Gradient Descent ---
    w = np.zeros(d)
    b = 0.0
    unclipped_losses = []
    for _ in range(n_epochs):
        r = X @ w + b - y
        gw = (2 / n) * (X.T @ r)
        gb = (2 / n) * np.sum(r)
        w -= lr * gw
        b -= lr * gb
        unclipped_losses.append(float(np.mean((X @ w + b - y) ** 2)))
        
    # --- 2. Element-wise Value Clipping ---
    w = np.zeros(d)
    b = 0.0
    clip_value_losses = []
    for _ in range(n_epochs):
        r = X @ w + b - y
        gw = (2 / n) * (X.T @ r)
        gb = (2 / n) * np.sum(r)
        
        gw_clipped = np.clip(gw, -clip_value, clip_value)
        gb_clipped = np.clip(gb, -clip_value, clip_value)
        
        w -= lr * gw_clipped
        b -= lr * gb_clipped
        clip_value_losses.append(float(np.mean((X @ w + b - y) ** 2)))
        
    # --- 3. Global Norm Clipping ---
    w = np.zeros(d)
    b = 0.0
    clip_norm_losses = []
    for _ in range(n_epochs):
        r = X @ w + b - y
        gw = (2 / n) * (X.T @ r)
        gb = (2 / n) * np.sum(r)
        
        # Concatenate weights and bias gradients into a single flat vector
        g = np.concatenate([gw.ravel(), [gb]])
        gnorm = np.linalg.norm(g)
        
        if gnorm > clip_norm:
            scale = clip_norm / gnorm
            gw_clipped = gw * scale
            gb_clipped = gb * scale
        else:
            gw_clipped = gw
            gb_clipped = gb
            
        w -= lr * gw_clipped
        b -= lr * gb_clipped
        clip_norm_losses.append(float(np.mean((X @ w + b - y) ** 2)))
        
    return {
        "unclipped_losses": unclipped_losses,
        "clip_value_losses": clip_value_losses,
        "clip_norm_losses": clip_norm_losses
    }
