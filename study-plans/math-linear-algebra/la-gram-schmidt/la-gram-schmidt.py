import numpy as np

def gram_schmidt(vectors: list) -> np.ndarray:
    """
    Returns a float64 array whose rows form the ordered orthonormal basis.
    """
    # Convert input list of lists to a NumPy float64 array
    V = np.array(vectors, dtype=np.float64)
    k, n = V.shape
    Q = np.zeros((k, n), dtype=np.float64)
    
    for i in range(k):
        # Step 1: Normalize the current residual vector
        norm = np.linalg.norm(V[i])
        Q[i] = V[i] / norm
        
        # Step 2: Subtract its projection from all remaining vectors
        for j in range(i + 1, k):
            V[j] -= np.dot(V[j], Q[i]) * Q[i]
            
    return Q
