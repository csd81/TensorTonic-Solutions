import numpy as np

def least_squares(A: list, b: list) -> np.ndarray:
    """
    Returns the float64 least-squares solution vector.
    """
    # Convert inputs to NumPy float64 arrays
    A_arr = np.array(A, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    
    # Compute the reduced QR decomposition (mode='reduced')
    Q, R = np.linalg.qr(A_arr, mode='reduced')
    
    # Compute the right-hand side vector: Q^T * b
    b_transformed = np.dot(Q.T, b_arr)
    
    # Solve the upper-triangular system R * x = Q^T * b directly
    return np.linalg.solve(R, b_transformed)
