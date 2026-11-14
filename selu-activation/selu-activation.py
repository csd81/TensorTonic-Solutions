import math

def selu(x: list) -> list:
    """
    Returns SELU values rounded to four decimal places.
    """
    LAMBDA = 1.0507009873554804934193349852946
    ALPHA = 1.6732632423543772848170429916717
    
    result = []
    for val in x:
        if val > 0:
            res = LAMBDA * val
        else:
            res = LAMBDA * ALPHA * (math.exp(val) - 1)
        result.append(round(float(res), 4))
        
    return result
