import numpy as np

def lu_decomposition(A: list) -> dict:
    """
    Returns a dictionary containing float64 arrays L and U.
    """
    A_arr = np.array(A, dtype=np.float64)
    n = A_arr.shape[0]
    
    # Initialize L as an identity matrix and U as zeros
    L = np.eye(n, dtype=np.float64)
    U = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        # 1. Compute entries for the i-th row of U
        for j in range(i, n):
            U[i, j] = A_arr[i, j] - np.dot(L[i, :i], U[:i, j])
            
        # 2. Compute entries for the i-th column of L
        for j in range(i + 1, n):
            # Avoid division by zero (guaranteed non-zero by problem constraints)
            L[j, i] = (A_arr[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]
            
    return {"L": L, "U": U}
