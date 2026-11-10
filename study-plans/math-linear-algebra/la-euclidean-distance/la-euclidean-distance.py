import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a float.
    """
    vec_x = np.array(x, dtype=np.float64)
    vec_y = np.array(y, dtype=np.float64)

    distance     = np.sqrt(np.sum( (vec_x - vec_y) ** 2 ))
    return float(distance)