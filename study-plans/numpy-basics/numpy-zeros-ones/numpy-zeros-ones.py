import numpy as np

def create_filled_array(shape: list, kind: str) -> np.ndarray:
    """
    Returns a 2D float64 array of zeros or ones with the requested shape.
    """
    shape_tuple = tuple(shape)

    if kind == "zeros":
        return np.zeros(shape_tuple, dtype=np.float64)
    elif kind == "ones":
        return np.ones(shape_tuple, dtype=np.float64)
    else:
        raise ValueError("Kind must be either 'zeros' or 'ones'")

