import numpy as np

def svd(A: list) -> dict:
    """
    Returns a dictionary containing float64 arrays U, s, and Vt.
    """
    # Convert the input list of lists into a NumPy float64 array
    A_arr = np.array(A, dtype=np.float64)
    
    # Compute the reduced SVD (full_matrices=False ensures shape matches k = min(m, n))
    U, s, Vt = np.linalg.svd(A_arr, full_matrices=False)
    
    # Return a dictionary with exactly the requested keys
    return {
        "U": U,
        "s": s,
        "Vt": Vt
    }
