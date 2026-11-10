import numpy as np

def impute(X: list, method: str = "mean") -> list:
    """
    Returns a new matrix with missing entries filled.
    """
    # Convert X to a copied NumPy float64 array (null converts to np.nan)
    arr = np.array(X, dtype=np.float64)
    
    # Iterate through each column
    for j in range(arr.shape[1]):
        col = arr[:, j]
        
        # Get only the observed (non-nan) values
        observed = col[~np.isnan(col)]
        
        # Determine the fill value based on constraints
        if len(observed) == 0:
            fill_value = 0.0
        elif method == "mean":
            fill_value = np.mean(observed)
        elif method == "median":
            fill_value = np.median(observed)
            
        # Replace the missing entries in the column
        col[np.isnan(col)] = fill_value
        
    # Round to 4 decimal places and return as a nested list
    return np.round(arr, 4).tolist()
