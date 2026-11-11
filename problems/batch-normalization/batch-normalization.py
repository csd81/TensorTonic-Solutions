import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    if x.ndim == 2:
        axes = 0
        shape = (1, -1)
    elif x.ndim == 4:
        axes = (0, 2, 3)
        shape = (1, -1, 1, 1)
    else:
        raise ValueError("Input x must be 2D or 4D.")

    # Compute mean and population variance over the specified axes
    mean = np.mean(x, axis=axes, keepdims=True)
    var = np.var(x, axis=axes, keepdims=True)

    # Normalize
    x_hat = (x - mean) / np.sqrt(var + eps)

    # Reshape gamma and beta for broadcasting, then scale and shift
    gamma = gamma.reshape(shape)
    beta = beta.reshape(shape)
    out = gamma * x_hat + beta

    return out
