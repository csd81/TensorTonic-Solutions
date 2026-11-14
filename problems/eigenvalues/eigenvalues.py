import numpy as np

def calculate_eigenvalues(matrix: list) -> np.ndarray:
    """
    Returns a sorted NumPy array of real eigenvalues.
    """
    # Convert the input list to a NumPy array
    A = np.asarray(matrix, dtype=float)
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    
    # Extract real parts (in case of 0j complex dtype) and sort in ascending order
    return np.sort(eigenvalues.real)
