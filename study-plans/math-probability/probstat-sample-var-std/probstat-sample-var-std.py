import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns sample variance and standard deviation as Python floats.
    """
    arr = np.asarray(x, dtype=np.float64) 
    variance = np.var(arr, ddof=1)
    std_dev = np.sqrt(variance)
    
    return {
        "variance": float(variance),
        "std_dev":  float(std_dev)
    }