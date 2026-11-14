import numpy as np

def recalibrate_after_downsampling(p: list, negative_keep_rate: float) -> np.ndarray:
    """
    Returns the corrected probabilities for a model trained after keeping only a fraction of the negatives.
    """
    arr = np.asarray(p, dtype=np.float64)
    
    if arr.size == 0 or negative_keep_rate == 1.0:
        return arr.copy()
        
    return arr / (arr + (1.0 - arr) / negative_keep_rate)
