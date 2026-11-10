import numpy as np

def eigendecompose(A: list) -> dict:
    """
    Returns sorted float64 eigenvalues and eigenvectors in a dictionary.
    """
    # Convert input list of lists to a numpy float64 array
    A_arr = np.array(A, dtype=np.float64)
    
    # Compute the eigenvalues and right eigenvectors
    # eigenvalues shape: (n,), eigenvectors shape: (n, n) where columns are eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A_arr)
    
    # Get sorting indices based on descending order of eigenvalue magnitudes (absolute values)
    # np.abs(eigenvalues) gets the magnitudes
    # np.argsort returns ascending indices; [::-1] reverses it to descending
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
    
    # Sort both the eigenvalues and the columns of the eigenvector matrix
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Return exactly the requested dictionary structure
    return {
        "eigenvalues": sorted_eigenvalues,
        "eigenvectors": sorted_eigenvectors
    }
