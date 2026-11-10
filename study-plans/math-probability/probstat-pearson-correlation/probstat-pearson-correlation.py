import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the float64 feature-correlation matrix.
    """
    data = np.asarray(X, dtype = np.float64)
    with np.errstate(divide = "ignore", invalid = "ignore"):
        return np.asarray(np.corrcoef(data, rowvar=False), dtype = np.float64)