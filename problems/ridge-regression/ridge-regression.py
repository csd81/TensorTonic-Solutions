import numpy as np

def ridge_regression(X: list, y: list, lam: float) -> list:
    """
    Returns the ridge-regression weight vector.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    n_features = X_arr.shape[1]
    A = X_arr.T @ X_arr + lam * np.eye(n_features, dtype=float)
    b = X_arr.T @ y_arr

    w = np.linalg.solve(A, b)
    return [float(val) for val in w]
