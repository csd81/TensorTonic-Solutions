import numpy as np

def adamw_compare(X: list, y: list, lr: float, beta1: float, beta2: float, weight_decay: float, n_epochs: int) -> dict:
    """
    Returns Adam with L2 and AdamW MSE loss curves in a dictionary.
    """
    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    
    n_samples, n_features = X_arr.shape
    eps = 1e-8
    
    # -------------------------------------------------------------------------
    # 1. Run Adam with L2 Regularization
    # -------------------------------------------------------------------------
    w_l2 = np.zeros(n_features, dtype=np.float64)
    m_l2 = np.zeros(n_features, dtype=np.float64)
    v_l2 = np.zeros(n_features, dtype=np.float64)
    adam_l2_losses = []
    
    for epoch in range(n_epochs):
        t = epoch + 1
        
        # Record loss before update
        preds = X_arr @ w_l2
        loss = float(np.mean((preds - y_arr) ** 2))
        adam_l2_losses.append(loss)
        
        # Pure data gradient
        grad_data = (2.0 / n_samples) * (X_arr.T @ (preds - y_arr))
        
        # Add L2 penalty to the gradient before moment tracking
        grad_total = grad_data + weight_decay * w_l2
        
        # Update moments and apply bias correction
        m_l2 = beta1 * m_l2 + (1.0 - beta1) * grad_total
        v_l2 = beta2 * v_l2 + (1.0 - beta2) * (grad_total ** 2)
        
        m_hat = m_l2 / (1.0 - beta1 ** t)
        v_hat = v_l2 / (1.0 - beta2 ** t)
        
        # Update weights (eps outside sqrt)
        w_l2 -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
    # -------------------------------------------------------------------------
    # 2. Run AdamW (Decoupled Weight Decay)
    # -------------------------------------------------------------------------
    w_w = np.zeros(n_features, dtype=np.float64)
    m_w = np.zeros(n_features, dtype=np.float64)
    v_w = np.zeros(n_features, dtype=np.float64)
    adamw_losses = []
    
    for epoch in range(n_epochs):
        t = epoch + 1
        
        # Record loss before update
        preds = X_arr @ w_w
        loss = float(np.mean((preds - y_arr) ** 2))
        adamw_losses.append(loss)
        
        # Pure data gradient
        grad_data = (2.0 / n_samples) * (X_arr.T @ (preds - y_arr))
        
        # Update moments on pure data gradient only
        m_w = beta1 * m_w + (1.0 - beta1) * grad_data
        v_w = beta2 * v_w + (1.0 - beta2) * (grad_data ** 2)
        
        m_hat = m_w / (1.0 - beta1 ** t)
        v_hat = v_w / (1.0 - beta2 ** t)
        
        # Update weights and apply decoupled decay directly
        w_w = w_w - lr * m_hat / (np.sqrt(v_hat) + eps) - lr * weight_decay * w_w
        
    return {
        "adam_l2_losses": adam_l2_losses,
        "adamw_losses": adamw_losses
    }
