import numpy as np

def low_rank_approximation(A: list, r: int) -> np.ndarray:
    """
    Returns the best rank-r float64 approximation.
    """
    # Convert input list of lists to a numpy float64 array
    A_arr = np.array(A, dtype=np.float64)
    
    # Compute the Reduced Singular Value Decomposition (SVD)
    U, s, Vt = np.linalg.svd(A_arr, full_matrices=False)
    
    # Truncate components to retain only the top r singular values/vectors
    U_r = U[:, :r]
    s_r = s[:r]
    Vt_r = Vt[:r, :]
    
    # Reconstruct the rank-r matrix product: U_r * diag(s_r) * Vt_r
    A_r = U_r @ np.diag(s_r) @ Vt_r
    
    return A_r
