import numpy as np

def rbf_kernel_matrix(X: list, gamma: float) -> np.ndarray:
    """
    Returns the float64 pairwise RBF kernel matrix.
    """
    # Convert input to a float64 NumPy array
    X_arr = np.array(X, dtype=np.float64)
    
    # Compute squared norms of each row vector: shape (n, 1)
    # Using the identity: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * <x_i, x_j>
    sq_norms = np.sum(X_arr**2, axis=1, keepdims=True)
    
    # Compute the full pairwise squared Euclidean distance matrix
    # sq_norms adds ||x_i||^2 row-wise, sq_norms.T adds ||x_j||^2 column-wise via broadcasting
    sq_dists = sq_norms + sq_norms.T - 2 * np.dot(X_arr, X_arr.T)
    
    # Clamp negative floating-point roundoff errors to zero
    sq_dists = np.maximum(sq_dists, 0.0)
    
    # Compute the radial basis function kernel
    K = np.exp(-gamma * sq_dists)
    
    return K
