import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    # Convert input list to a numpy array of floats
    X_arr = np.array(X, dtype=float)
    
    # 1. Center the data by subtracting the column means
    mean = np.mean(X_arr, axis=0)
    X_centered = X_arr - mean
    
    # 2. Compute the d x d covariance matrix using sample covariance (divide by n-1)
    n = X_arr.shape[0]
    C = np.dot(X_centered.T, X_centered) / (n - 1)
    
    # 3. Find eigenvalues and eigenvectors of C
    # np.linalg.eigh is optimized and stable for symmetric matrices like covariance
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Select the top-k eigenvectors (principal components)
    W = eigenvectors[:, :k]
    
    # 4. Project the centered data onto these k eigenvectors
    X_proj = np.dot(X_centered, W)
    
    # Return as an n x k list of floats
    return X_proj.tolist()
