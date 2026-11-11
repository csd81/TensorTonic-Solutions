import numpy as np


def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)

    # Filter for indices where p > 0 (since terms with p_i = 0 contribute 0)
    mask = p_arr > 0
    p_pos = p_arr[mask]
    q_pos = np.maximum(q_arr[mask], eps)

    # Compute KL divergence and convert NumPy scalar to standard Python float
    divergence = np.sum(p_pos * np.log(p_pos / q_pos))
    return float(divergence)
