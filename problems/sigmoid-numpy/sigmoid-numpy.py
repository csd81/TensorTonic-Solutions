import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    is_scalar = np.isscalar(x)
    arr = np.asarray(x, dtype=float)
    out = np.where(
        arr >= 0,
        1.0 / (1.0 + np.exp(-arr)),
        np.exp(arr) / (1.0 + np.exp(arr)),
    )
    if is_scalar:
        return float(out)
    return out
