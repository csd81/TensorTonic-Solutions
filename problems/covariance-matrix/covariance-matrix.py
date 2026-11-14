import numpy as np


def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    X_arr = np.asarray(X, dtype=float)
    N = X_arr.shape[0]

    # Center each feature (column-wise mean)
    mean = np.mean(X_arr, axis=0)
    X_c = X_arr - mean

    # Compute sample covariance matrix: (X_c^T @ X_c) / (N - 1)
    cov = (X_c.T @ X_c) / (N - 1)
    return cov
