import numpy as np

def pca_project(X: list, n_components: int) -> np.ndarray:
    """
    Returns the projected samples as a float64 array.
    """
    # Convert input list to a numpy float64 array
    X = np.array(X, dtype=np.float64)
    n, d = X.shape
    
    # 1. Center every feature using its sample mean
    mu = np.mean(X, axis=0)
    X_c = X - mu
    
    # 2. Form the sample covariance matrix with denominator (n - 1)
    C = (X_c.T @ X_c) / (n - 1)
    
    # 3. Eigendecompose the sample covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # 4. Sort eigenvalues and corresponding eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # 5. Select the top k eigenvectors
    V_k = eigenvectors[:, :n_components]
    
    # 6. Project the centered data onto the leading components
    Z = X_c @ V_k
    
    return Z.astype(np.float64)
