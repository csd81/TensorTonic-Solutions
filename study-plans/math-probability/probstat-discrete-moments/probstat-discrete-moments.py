import numpy as np

def discrete_moments(values: list, probabilities: list) -> list:
    """
    Returns the first moment, second moment, variance, and standard deviation.
    """
    
    x = np.array(values)
    p = np.array(probabilities)
    mean = float(np.sum(x*p))
    second_moment = float(np.sum((x**2) * p))
    variance = second_moment - (mean**2)
    if variance < 0:
        variance = 0.0

    std_dev = float(np.sqrt(variance))
    
    # Return all metrics rounded to 4 decimal places    
    return [
        round(mean, 4),
        round(second_moment, 4),
        round(variance, 4),
        round(std_dev, 4)
    ]
