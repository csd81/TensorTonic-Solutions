import numpy as np

def dropout(X: list, mask: list, drop_prob: float, mode: str) -> list:
    """
    Returns the train-mode dropout result or unchanged test input.
    """
    # 1. Convert X and mask to NumPy float64 arrays
    X_arr = np.array(X, dtype=np.float64)
    
    if mode == "train":
        mask_arr = np.array(mask, dtype=np.float64)
        # Calculate keep probability
        keep_prob = 1.0 - drop_prob
        
        # Apply inverted dropout formula: Y = (X * M) / (1 - p)
        # Avoid division by zero if drop_prob is 0 (though constraints state drop_prob < 1)
        Y_arr = (X_arr * mask_arr) / keep_prob
    else:
        # During test mode, return X unchanged and ignore both mask and drop_prob
        Y_arr = X_arr
        
    # 2. Round numeric outputs to 4 decimal places and convert back to nested lists
    return np.round(Y_arr, 4).tolist()
