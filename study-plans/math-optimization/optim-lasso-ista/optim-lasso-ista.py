import numpy as np

def lasso_ista(X: list, y: list, lam: float, n_iters: int) -> list:
    """
    Returns the complete ISTA weight trajectory as a list of lists.
    """
    # Convert inputs to numpy arrays
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)
    n, d = X_arr.shape
    
    # 1. Compute the step size t
    XTX_n = (X_arr.T @ X_arr) / n
    eigenvalues = np.linalg.eigvalsh(XTX_n)
    max_eig = np.max(eigenvalues)
    t = 1.0 / max_eig
    
    # Threshold parameter tau
    tau = lam * t
    
    # Initialize weights to zero
    w = np.zeros(d)
    
    # Track the weight trajectory (include the initial zero weights)
    trajectory = [w.tolist()]
    
    # 2. Iterative Optimization Loop
    for _ in range(n_iters):
        # Compute gradient step on the squared error term
        grad = (X_arr.T @ (X_arr @ w - y_arr)) / n
        w_next = w - t * grad
        
        # Apply elementwise soft-thresholding
        w = np.sign(w_next) * np.maximum(np.abs(w_next) - tau, 0.0)
        
        # Append copy of current weights to trajectory
        trajectory.append(w.tolist())
        
    return trajectory
