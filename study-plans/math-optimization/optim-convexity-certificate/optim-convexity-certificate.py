import numpy as np

def convexity_certificate(H: list) -> dict:
    """
    Returns the PSD decision and smallest eigenvalue in a dictionary.
    """
    # Convert the list of lists into a numpy array
    matrix = np.array(H, dtype=float)
    
    # Compute the eigenvalues using the symmetric eigensolver
    # eigh returns them in ascending order
    eigenvalues = np.linalg.eigh(matrix)[0]
    
    # Extract the minimum eigenvalue
    min_eigen = float(eigenvalues[0])
    
    # Check against the numerical tolerance threshold (-1e-6)
    is_convex = min_eigen >= -1e-6
    
    return {
        "is_convex": is_convex,
        "min_eigenvalue": round(min_eigen, 6)
    }
