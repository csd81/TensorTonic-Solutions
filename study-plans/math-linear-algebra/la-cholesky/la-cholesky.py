import numpy as np

def cholesky_decompose(A: list):
    """
    Returns a float64 lower-triangular array, or None.
    """
    A = np.array(A, dtype=np.float64)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                val = A[i, i] - s
                if val <= 0:
                    return None
                L[i, j] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
                
    return L
