import numpy as np

def linear_warmup(X: list, y: list, base_lr: float, warmup_epochs: int, total_epochs: int) -> dict:
    """
    Returns the learning-rate schedule and pre-update losses.
    """
    X = np.array(X)
    y = np.array(y)
    
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    
    lr_schedule = []
    losses = []
    
    for e in range(total_epochs):
        # 1. Compute current predictions and MSE loss
        y_pred = X @ w
        mse_loss = np.mean((y - y_pred) ** 2)
        losses.append(float(mse_loss))
        
        # 2. Calculate learning rate for the current epoch
        lr = base_lr * min((e + 1) / warmup_epochs, 1.0)
        lr_schedule.append(float(lr))
        
        # 3. Compute gradient and update weights
        # Gradient of MSE: (2 / N) * X^T * (Xw - y)
        gradient = (2 / n_samples) * X.T @ (y_pred - y)
        w -= lr * gradient
        
    return {
        "lr_schedule": lr_schedule,
        "losses": losses
    }
