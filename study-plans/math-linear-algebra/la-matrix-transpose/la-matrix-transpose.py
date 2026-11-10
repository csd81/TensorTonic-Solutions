import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transpose as a float64 array.
    """
    # Get dimensions of the original matrix
    m = len(A)
    n = len(A[0])
    
    # Create the transposed grid using list comprehension
    # Outer loop iterates through columns (j), inner loop iterates through rows (i)
    transposed = [[A[i][j] for i in range(m)] for j in range(n)]
    
    # Return as a NumPy float64 array
    return np.array(transposed, dtype=np.float64)
