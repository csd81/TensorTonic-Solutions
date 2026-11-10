import numpy as np

def matrix_rank(A: list) -> int:
    """
    Returns the numerical rank as a Python integer.
    """
    # Convert input to a float64 numpy array
    matrix = np.array(A, dtype=np.float64)
    m, n = matrix.shape
    
    # Compute the singular values using SVD
    # compute_uv=False avoids unnecessary calculation of U and Vt matrices
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    
    # If the matrix is empty or contains only zeros, rank is 0
    if len(singular_values) == 0 or np.all(singular_values == 0):
        return 0
        
    # Find the maximum singular value
    sigma_max = np.max(singular_values)
    
    # Machine precision for float64 (approx 2.22e-16)
    epsilon = np.finfo(np.float64).eps
    
    # Compute the NumPy default tolerance formula
    tau = max(m, n) * sigma_max * epsilon
    
    # Count how many singular values are strictly greater than tau
    rank = int(np.sum(singular_values > tau))
    
    return rank
