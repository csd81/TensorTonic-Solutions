import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    n_samples, n_features = X_arr.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    for _ in range(steps):
        z = X_arr @ w + b
        p = _sigmoid(z)
        error = p - y_arr

        dw = (X_arr.T @ error) / n_samples
        db = float(np.mean(error))

        w -= lr * dw
        b -= lr * db

    return w, float(b)
