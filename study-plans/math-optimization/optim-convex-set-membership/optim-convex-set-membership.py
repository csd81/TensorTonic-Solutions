import numpy as np

def convex_set_membership(A: list, b: list, x: list) -> dict:
    """
    Returns the membership decision and maximum violation in a dictionary.
    """
    # Convert lists to NumPy arrays for vectorized operations
    A_arr = np.array(A, dtype=float)
    b_arr = np.array(b, dtype=float)
    x_arr = np.array(x, dtype=float)
    
    # Compute all constraint residuals: Ax - b
    residuals = A_arr @ x_arr - b_arr
    
    # Find the maximum constraint violation
    max_violation = float(np.max(residuals))
    
    # Determine membership using the 0.000001 tolerance threshold
    in_set = max_violation <= 1e-6
    
    return {
        "in_set": in_set,
        "max_violation": round(max_violation, 6)
    }
