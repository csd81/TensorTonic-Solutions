import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    values = np.asarray(x, dtype=np.float64)
    deviations = values - np.mean(values)
    variance = np.sum(deviations**2) / (values.size -1)

    return {
        "variance": float(variance),
        "standard_deviation": float(np.sqrt(variance))
        
    }