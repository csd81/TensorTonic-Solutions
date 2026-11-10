import numpy as np

def pca(X: list, n_components: int = 2) -> tuple:
    """
    Returns transformed rows and explained variance ratios.
    """
    # 1. Convert X to a NumPy float64 array
    X = np.array(X, dtype=np.float64)
    n, d = X.shape
    
    # 2. Center every feature (column)
    X_mean = np.mean(X, axis=0)
    X_c = X - X_mean
    
    # 3. Form the unbiased covariance matrix
    # Sigma = 1 / (n - 1) * X_c^T * X_c
    Sigma = np.dot(X_c.T, X_c) / (n - 1)
    
    # 4. Compute eigenvalues and eigenvectors using np.linalg.eigh
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    
    # 5. Stably sort eigenvalues and eigenvectors in descending order
    # argsort returns indices to sort in ascending order, [::-1] reverses it
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 6. Canonicalize eigenvector signs before projection
    # For every eigenvector (column), find the element with the largest absolute value.
    # If its value is negative, flip the sign of the whole column.
    for j in range(eigenvectors.shape[1]):
        ev = eigenvectors[:, j]
        max_abs_idx = np.argmax(np.abs(ev))
        if ev[max_abs_idx] < 0:
            eigenvectors[:, j] = -ev
            
    # 7. Project onto the first n_components columns
    W = eigenvectors[:, :n_components]
    X_transformed = np.dot(X_c, W)
    
    # 8. Calculate the explained variance ratio
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues[:n_components] / total_variance
    
    # 9. Round numeric outputs to 4 decimal places
    return (
        np.round(X_transformed, 4).tolist(), 
        np.round(explained_variance_ratio, 4).tolist()
    )
