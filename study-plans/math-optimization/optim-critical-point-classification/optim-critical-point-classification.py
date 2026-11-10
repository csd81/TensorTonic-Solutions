import numpy as np

def classify_critical_point(H: list) -> str:
    """
    Returns the second-order classification string.
    """
    # Convert list of lists to a numpy array
    matrix = np.array(H, dtype=float)
    
    # Compute eigenvalues for a symmetric matrix
    eigenvalues = np.linalg.eigh(matrix)[0]
    
    # Define tolerance threshold
    tol = 1e-6
    
    has_pos = False
    has_neg = False
    
    for lam in eigenvalues:
        if abs(lam) <= tol:
            return "degenerate"
        elif lam > tol:
            has_pos = True
        elif lam < -tol:
            has_neg = True
            
    if has_pos and has_neg:
        return "saddle"
    elif has_pos:
        return "local_min"
    else:
        return "local_max"
