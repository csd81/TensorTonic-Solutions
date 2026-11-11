import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    x_arr = np.asarray(x)
    pmf = np.where(x_arr == 1, p, 1.0 - p)
    mean = float(p)
    variance = float(p * (1.0 - p))
    
    return {
        "pmf": pmf,
        "mean": mean,
        "variance": variance,
    }
