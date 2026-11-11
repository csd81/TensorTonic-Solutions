import math


def he_initialization(W: list, fan_in: int) -> list:
    """
    Returns the weights mapped to the He uniform range.
    """
    limit = math.sqrt(6.0 / fan_in)
    return [[round(val * 2 * limit - limit, 4) for val in row] for row in W]
