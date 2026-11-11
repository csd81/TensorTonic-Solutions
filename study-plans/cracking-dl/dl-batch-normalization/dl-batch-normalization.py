import numpy as np

def batch_norm(X: list, gamma: list, beta: list, running_mean: list, running_var: list, mode: str) -> dict:
    """
    Returns normalized output and resulting running statistics.
    """
    # Convert lists to NumPy arrays with float64 precision
    X_np = np.array(X, dtype=np.float64)
    gamma_np = np.array(gamma, dtype=np.float64)
    beta_np = np.array(beta, dtype=np.float64)
    
    # Explicitly copy running statistics to satisfy requirements
    rm_np = np.array(running_mean, dtype=np.float64).copy()
    rv_np = np.array(running_var, dtype=np.float64).copy()
    
    eps = 1e-5
    
    if mode == "train":
        # Calculate current batch mean and variance across the batch dimension (axis 0)
        mu_B = np.mean(X_np, axis=0)
        var_B = np.var(X_np, axis=0, ddof=0)
        
        # Normalize and scale
        X_hat = (X_np - mu_B) / np.sqrt(var_B + eps)
        output = gamma_np * X_hat + beta_np
        
        # Update running statistics using momentum 0.1
        rm_np = 0.9 * rm_np + 0.1 * mu_B
        rv_np = 0.9 * rv_np + 0.1 * var_B
    else:
        # Test mode uses pre-calculated running statistics directly
        X_hat = (X_np - rm_np) / np.sqrt(rv_np + eps)
        output = gamma_np * X_hat + beta_np
        
    # Return formatted dictionary rounded to 4 decimal places
    return {
        "output": np.round(output, 4).tolist(),
        "running_mean": np.round(rm_np, 4).tolist(),
        "running_var": np.round(rv_np, 4).tolist()
    }
