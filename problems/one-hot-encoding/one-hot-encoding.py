import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    y_arr = np.asarray(y, dtype=int)
    
    if num_classes is None:
        num_classes = np.max(y_arr) + 1
        
    N = y_arr.shape[0]
    out = np.zeros((N, num_classes), dtype=float)
    out[np.arange(N), y_arr] = 1.0
    
    return out
