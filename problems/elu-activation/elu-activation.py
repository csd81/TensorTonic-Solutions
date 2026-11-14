import math


def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """
    return [float(val) if val > 0 else float(alpha * math.expm1(val)) for val in x]
