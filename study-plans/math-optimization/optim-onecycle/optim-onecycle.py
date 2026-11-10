import numpy as np

def onecycle_train(X: list, y: list, max_lr: float, div_factor: float, final_div: float, pct_start: float, total_epochs: int) -> dict:
    """
    Returns the one-cycle schedule, losses, initial rate, and minimum rate.
    """
    X = np.array(X)
    y = np.array(y)
    
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    
    # 1. Compute bounding learning rates
    initial_lr = max_lr / div_factor
    min_lr = initial_lr / final_div
    
    # 2. Determine peak index with clamping constraints
    peak_index = int(total_epochs * pct_start)
    peak_index = max(1, min(peak_index, total_epochs - 2))
    
    lr_schedule = []
    losses = []
    
    for e in range(total_epochs):
        # 3. Compute current predictions and MSE loss
        y_pred = X @ w
        mse_loss = np.mean((y - y_pred) ** 2)
        losses.append(float(mse_loss))
        
        # 4. Interpolate learning rate based on current phase
        if e <= peak_index:
            # Phase 1: Cosine annealing from initial_lr up to max_lr
            progress = e / peak_index
            cos_arg = np.pi * (1.0 - progress)
            lr = initial_lr + 0.5 * (max_lr - initial_lr) * (1.0 + np.cos(cos_arg))
        else:
            # Phase 2: Cosine annealing from max_lr down to min_lr
            progress = (e - peak_index) / (total_epochs - 1 - peak_index)
            cos_arg = np.pi * progress
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + np.cos(cos_arg))
            
        lr_schedule.append(float(lr))
        
        # 5. Compute gradient and update weights
        gradient = (2.0 / n_samples) * X.T @ (y_pred - y)
        w -= lr * gradient
        
    return {
        "lr_schedule": lr_schedule,
        "losses": losses,
        "initial_lr": initial_lr,
        "min_lr": min_lr
    }
