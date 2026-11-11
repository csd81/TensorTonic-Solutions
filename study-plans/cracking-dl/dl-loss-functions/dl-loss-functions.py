import numpy as np

def loss_functions(y_true: list, y_pred: list, loss_type: str) -> float:
    """
    Returns the selected mean loss as a float, rounded to 4 decimal places.
    """
    # Convert predictions to a NumPy float64 array
    y_pred_arr = np.array(y_pred, dtype=np.float64)
    y_true_arr = np.array(y_true)
    
    if loss_type == "mse":
        # Mean Squared Error = mean((y - y_hat)^2)
        loss = np.mean((y_true_arr - y_pred_arr) ** 2)
        
    elif loss_type == "bce":
        # Clip BCE probabilities to [1e-15, 1 - 1e-15] for numerical stability
        eps = 1e-15
        p = np.clip(y_pred_arr, eps, 1.0 - eps)
        # Binary Cross-Entropy = -mean([y * log(p) + (1 - y) * log(1 - p)])
        loss = -np.mean(y_true_arr * np.log(p) + (1.0 - y_true_arr) * np.log(1.0 - p))
        
    elif loss_type == "cce":
        # Categorical Cross-Entropy (y_true contains class indices, y_pred contains logits)
        # Implement stable log-sum-exp: log(sum(exp(x))) = max_x + log(sum(exp(x - max_x)))
        max_logits = np.max(y_pred_arr, axis=-1, keepdims=True)
        stable_logits = y_pred_arr - max_logits
        
        sum_exp = np.sum(np.exp(stable_logits), axis=-1, keepdims=True)
        log_sum_exp = max_logits + np.log(sum_exp)
        
        # Log-softmax for each logit matrix entry: log_softmax = logit - log_sum_exp
        # Extract the logit corresponding to the true class index for each sample
        num_samples = len(y_true_arr)
        correct_class_logits = y_pred_arr[np.arange(num_samples), y_true_arr]
        correct_class_log_sum_exp = log_sum_exp.flatten()
        
        # Loss per sample = log_sum_exp - correct_class_logit
        sample_losses = correct_class_log_sum_exp - correct_class_logits
        loss = np.mean(sample_losses)
        
    elif loss_type == "hinge":
        # Hinge Loss = mean(max(0, 1 - y * y_hat))
        loss = np.mean(np.maximum(0.0, 1.0 - y_true_arr * y_pred_arr))
        
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
        
    return float(np.round(loss, 4))
