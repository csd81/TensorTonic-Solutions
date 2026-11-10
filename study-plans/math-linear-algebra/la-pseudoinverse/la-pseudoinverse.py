import numpy as np

def pseudoinverse(A: list) -> np.ndarray:
    """
    Returns the Moore-Penrose pseudoinverse as a float64 array.
    """
    # Convert input to float64 numpy array
    A_arr = np.array(A, dtype=np.float64)
    
    # Compute the full or reduced SVD
    # s is a 1D array of singular values sorted in descending order
    U, s, Vt = np.linalg.svd(A_arr, full_matrices=False)
    
    # Define tolerance threshold
    tol = 1e-10
    
    # Create the reciprocal singular values matrix (Sigma_plus)
    # Reciprocate values greater than tolerance, otherwise keep as 0
    s_plus = np.where(s > tol, 1.0 / s, 0.0)
    
    # Compute A_plus = V * Sigma_plus * U^T
    # Note: np.linalg.svd returns Vt (V transpose), so V is Vt.T
    # Since Sigma_plus is just a diagonal matrix vector representation, 
    # multiplying V by Sigma_plus can be written efficiently as V * s_plus
    A_plus = np.dot(Vt.T * s_plus, U.T)
    
    return A_plus
