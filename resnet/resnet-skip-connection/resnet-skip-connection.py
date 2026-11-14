import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through residual Jacobians.
    """
    g = np.array(x, dtype=np.float64, copy=True)
    for J in gradients_F:
        J_arr = np.asarray(J, dtype=np.float64)
        g = g + g @ J_arr
    return g

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through plain Jacobians.
    """
    g = np.array(x, dtype=np.float64, copy=True)
    for J in gradients_F:
        J_arr = np.asarray(J, dtype=np.float64)
        g = g @ J_arr
    return g
