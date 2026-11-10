import numpy as np
from scipy.stats import norm

def clt_confidence_interval(data: list, confidence: float) -> list:
    """
    Returns the sample mean, standard error, and confidence-interval endpoints.
    """
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data,ddof=1)
    se = std_dev / np.sqrt(n)
    z = norm.ppf((1+confidence)/2)
    lower_bound=mean-z*se
    upper_bound=mean+z*se
    # Return the four results rounded to 4 decimal places
    return [
        round(float(mean), 4),
        round(float(se), 4),
        round(float(lower_bound), 4),
        round(float(upper_bound), 4)
    ]
