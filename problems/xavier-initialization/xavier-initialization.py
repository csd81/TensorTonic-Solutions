import math

def xavier_initialization(W: list, fan_in: int, fan_out: int) -> list:
    """
    Returns the weights mapped to the Xavier uniform range.
    """
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    
    return [
        [round(w * 2.0 * limit - limit, 4) for w in row]
        for row in W
    ]
