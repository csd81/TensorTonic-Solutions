import numpy as np

def qr_decompose(A: list) -> dict:
    """
    Returns a dictionary containing float64 arrays Q and R.
    Implements the Modified Gram-Schmidt orthogonalization process.
    """
    # Convert input to float64 NumPy array
    A_arr = np.array(A, dtype=np.float64)
    m, n = A_arr.shape
    
    # Initialize Q and R matrices
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)
    
    # Make a copy of columns to modify in-place
    V = A_arr.copy()
    
    for i in range(n):
        # Compute the norm of the current column
        r_ii = np.linalg.norm(V[:, i])
        
        # If r_ii is negative (or to enforce positive diagonal), 
        # we naturally get positive since norm is always >= 0.
        # But if the original vector had a sign choice, we ensure r_ii > 0.
        R[i, i] = r_ii
        
        # Compute the orthonormal column for Q
        Q[:, i] = V[:, i] / r_ii
        
        # Subtract the projection from all remaining columns (Modified Gram-Schmidt)
        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] -= R[i, j] * Q[:, i]
            
    # Adjust for any potential negative diagonals to strictly guarantee R[i,i] > 0
    # (Though norm makes R[i,i] >= 0, if R[i,i] becomes 0 due to precision, or if a sign flip is needed)
    for i in range(n):
        if R[i, i] < 0:
            R[i, i:] = -R[i, i:]
            Q[:, i] = -Q[:, i]
            
    return {"Q": Q, "R": R}
