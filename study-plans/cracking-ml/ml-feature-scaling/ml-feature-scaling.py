import numpy as np

def feature_scale(X: list, method: str = "minmax") -> list:
    """
    Returns the independently scaled feature columns.
    """
    # Convert input to a float64 NumPy array
    X_arr = np.array(X, dtype=np.float64)
    
    if method == "minmax":
        # Compute min and max along axis 0
        min_vals = np.min(X_arr, axis=0)
        max_vals = np.max(X_arr, axis=0)
        denom = max_vals - min_vals
        
        # Replace zero denominators with one
        denom[denom == 0] = 1.0
        
        # Scale values
        scaled = (X_arr - min_vals) / denom
        
    elif method == "standard":
        # Compute mean and standard deviation (ddof=0) along axis 0
        mean_vals = np.mean(X_arr, axis=0)
        std_vals = np.std(X_arr, axis=0, ddof=0)
        denom = std_vals
        
        # Replace zero denominators with one
        denom[denom == 0] = 1.0
        
        # Scale values
        scaled = (X_arr - mean_vals) / denom
        
    else:
        raise ValueError("Method must be either 'minmax' or 'standard'")
        
    # Round to 4 decimal places and return as a nested list
    return np.round(scaled, 4).tolist()
