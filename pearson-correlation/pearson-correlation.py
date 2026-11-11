import numpy as np


def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    arr = np.asarray(X, dtype=np.float64)
    n = arr.shape[0]

    # Center each feature column by subtracting its mean
    centered = arr - np.mean(arr, axis=0)

    # Compute the sample covariance matrix (D, D) using Bessel's correction
    cov = (centered.T @ centered) / (n - 1)

    # Extract feature standard deviations from the diagonal of the covariance matrix
    std = np.sqrt(np.diag(cov))

    # Build the denominator matrix via outer product of standard deviations
    denom = np.outer(std, std)

    # Divide covariance by denominator, leaving NaN where denominator is zero
    corr = np.full_like(cov, np.nan)
    valid = denom > 0.0
    corr[valid] = cov[valid] / denom[valid]

    return corr
