import numpy as np

def hessian_convexity_analysis(X: list, y: list) -> dict:
    """
    Returns the half-MSE Hessian, eigenvalues, and convexity.
    """
    X_arr = np.array(X, dtype=float)
    n = X_arr.shape[0]
    
    # 1. Compute Hessian: H = (1/n) * X^T * X
    H = (1.0 / n) * (X_arr.T @ X_arr)
    
    # 2. Get eigenvalues in ascending order
    eigenvalues = np.linalg.eigvalsh(H)
    
    # 3. Check convexity against tolerance threshold
    tolerance = -0.0000000001
    is_convex = bool(np.all(eigenvalues >= tolerance))
    
    return {
        "hessian": H.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "is_convex": is_convex
    }
