import numpy as np

def geometric_pmf_mean(k: list, p: float) -> dict:
    """
    Returns a dictionary with pmf and mean.
    """
    k_arr = np.asarray(k, dtype=np.float64)
    pmf = ((1 - p) ** (k_arr - 1)) * p
    mean = float(1.0 / p)
    
    return {
        "pmf": pmf,
        "mean": mean
    }
