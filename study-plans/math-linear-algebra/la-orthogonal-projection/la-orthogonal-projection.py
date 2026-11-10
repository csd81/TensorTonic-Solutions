import numpy as np

def projection_matrix(A: list) -> np.ndarray:
    """
    Returns the float64 projector onto the column space of A.
    """
    # Convert input list of lists to a numpy float64 array
    A_arr = np.array(A, dtype=np.float64)
    
    # Compute the Gram matrix: A^T @ A
    Gram = A_arr.T @ A_arr
    
    # Solve the system (A^T @ A) * X = A^T instead of computing an explicit inverse.
    # This solves for X = (A^T @ A)^-1 @ A^T
    X = np.linalg.solve(Gram, A_arr.T)
    
    # Complete the projection matrix P = A @ X
    P = A_arr @ X
    
    return P
