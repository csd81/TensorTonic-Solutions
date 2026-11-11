import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    x_arr = np.asarray(x, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    return float(np.sum(x_arr * p_arr))
