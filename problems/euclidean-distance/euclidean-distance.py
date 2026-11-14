import numpy as np


def euclidean_distance(x: list, y: list) -> float:
    """Returns the Euclidean distance as a Python float."""
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # Calculate the square root of the sum of squared differences
    distance = np.sqrt(np.sum((x_arr - y_arr) ** 2))

    # Alternatively, using NumPy's built-in linear algebra norm:
    # distance = np.linalg.norm(x_arr - y_arr)

    return float(distance)
