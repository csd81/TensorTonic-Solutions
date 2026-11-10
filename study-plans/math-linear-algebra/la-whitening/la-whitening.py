import numpy as np

def whiten(X: list) -> np.ndarray:
    """
    Returns the whitened samples as a float64 array.
    """
    # Convert input to a float64 NumPy array
    X = np.array(X, dtype=np.float64)
    n, d = X.shape
    
    # 1. Center every feature using its sample mean
    mu = np.mean(X, axis=0)
    X_c = X - mu
    
    # 2. Form the sample covariance matrix with a denominator of n - 1
    # np.cov returns a 0-D array if d=1, so we cast it to 2D
    C = np.cov(X, rowvar=False, ddof=1)
    if d == 1:
        C = np.array([[C]])
        
    # 3. Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(C)
    
    # Order covariance eigenvectors by descending eigenvalue
    descending_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[descending_idx]
    eigvecs = eigvecs[:, descending_idx]
    
    # Orient each eigenvector so its largest-magnitude coordinate is positive
    for j in range(d):
        v = eigvecs[:, j]
        max_abs_idx = np.argmax(np.abs(v))
        if v[max_abs_idx] < 0:
            eigvecs[:, j] = -v
            
    # 4. Use reciprocal square-root scaling only for eigenvalues > 1e-10
    inv_sqrt_D = np.zeros(d, dtype=np.float64)
    for i in range(d):
        if eigvals[i] > 1e-10:
            inv_sqrt_D[i] = 1.0 / np.sqrt(eigvals[i])
            
    # 5. Transform the data: X_w = X_c @ V @ D^{-1/2}
    W = eigvecs @ np.diag(inv_sqrt_D)
    X_w = X_c @ W
    
    return X_w
