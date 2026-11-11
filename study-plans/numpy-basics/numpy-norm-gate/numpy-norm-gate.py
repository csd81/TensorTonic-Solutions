import numpy as np

def norm_gate(X: list, W: list, threshold: float) -> np.ndarray:
    """
    Returns an (n, k) float64 matrix of norm-gated transformed rows.
    """
    # Convert inputs to float64 NumPy arrays
    X_arr = np.array(X, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Compute the linear transform Z = X @ W
    Z = X_arr @ W_arr
    
    # Compute the L2 norm for each row (axis=1)
    row_norms = np.linalg.norm(Z, axis=1)
    
    # Create a boolean mask where the norm is less than the threshold
    mask = row_norms < threshold
    
    # Copy Z to Y and zero out the rows that do not pass the gate
    Y = Z.copy()
    Y[mask] = 0.0
    
    return Y
