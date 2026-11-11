import numpy as np


def gini_impurity(y_left: list, y_right: list) -> float:
    """
    Returns the impurity as a float.
    """

    def node_gini(y: list) -> float:
        arr = np.asarray(y)
        if arr.size == 0:
            return 0.0
        _, counts = np.unique(arr, return_counts=True)
        probs = counts / arr.size
        return float(1.0 - np.sum(probs**2))

    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    if n_total == 0:
        return 0.0

    g_left = node_gini(y_left)
    g_right = node_gini(y_right)

    weighted_gini = (n_left / n_total) * g_left + (n_right / n_total) * g_right
    return float(weighted_gini)
