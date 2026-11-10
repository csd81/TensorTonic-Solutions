import numpy as np

def linear_regression_gradient(X: list, y: list, w: list) -> dict:
    """
    Returns the SSE loss and analytical and numerical gradients.
    """
    # 1. Convert all inputs to NumPy float64 arrays
    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    w_arr = np.array(w, dtype=np.float64)
    
    # Helper function to compute the unnormalized sum of squared errors
    def compute_loss(weights):
        residuals = X_arr @ weights - y_arr
        return np.sum(residuals ** 2)
    
    # 2. Compute the current loss
    loss_val = compute_loss(w_arr)
    
    # 3. Compute analytical gradient: ∇L(w) = 2 * X^T * (Xw - y)
    analytical_grad = 2.0 * X_arr.T @ (X_arr @ w_arr - y_arr)
    
    # 4. Compute numerical gradient via central differences (h = 1e-5)
    h = 0.00001
    numerical_grad = np.zeros_like(w_arr)
    
    for i in range(len(w_arr)):
        # Perturb coordinate i positively
        w_plus = w_arr.copy()
        w_plus[i] += h
        loss_plus = compute_loss(w_plus)
        
        # Perturb coordinate i negatively
        w_minus = w_arr.copy()
        w_minus[i] -= h
        loss_minus = compute_loss(w_minus)
        
        # Central difference formula
        numerical_grad[i] = (loss_plus - loss_minus) / (2.0 * h)
        
    return {
        "loss": float(loss_val),
        "analytical_gradient": [float(x) for x in analytical_grad],
        "numerical_gradient": [float(x) for x in numerical_grad]
    }
