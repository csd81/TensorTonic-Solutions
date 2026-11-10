import numpy as np

def cosine_restarts(X: list, y: list, eta_max: float, eta_min: float, T_0: int, T_mult: int, total_epochs: int) -> dict:
    """
    Returns the cosine-restart schedule and pre-update losses.
    """
    X = np.array(X)
    y = np.array(y)
    
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    
    lr_schedule = []
    losses = []
    
    # Track current cycle duration and current position inside that cycle
    T_i = T_0
    T_cur = 0
    
    for e in range(total_epochs):
        # 1. Compute current predictions and MSE loss
        y_pred = X @ w
        mse_loss = np.mean((y - y_pred) ** 2)
        losses.append(float(mse_loss))
        
        # 2. Calculate learning rate using the SGDR formula
        cos_arg = np.pi * T_cur / T_i
        lr = eta_min + 0.5 * (eta_max - eta_min) * (1.0 + np.cos(cos_arg))
        lr_schedule.append(float(lr))
        
        # 3. Compute gradient and update weights
        gradient = (2.0 / n_samples) * X.T @ (y_pred - y)
        w -= lr * gradient
        
        # 4. Advance cycle tracking metrics
        T_cur += 1
        if T_cur == T_i:
            T_cur = 0
            T_i *= T_mult
            
    return {
        "lr_schedule": lr_schedule,
        "losses": losses
    }
