import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    y_arr = np.asarray(y)
    
    # Handle empty node
    if y_arr.size == 0:
        return 0.0
    
    # Get counts of each unique class present
    _, counts = np.unique(y_arr, return_counts=True)
    
    # Calculate probabilities (only non-zero counts are returned by np.unique)
    probabilities = counts / y_arr.size
    
    # Compute Shannon entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    # Cast to standard Python float (and avoid -0.0 on pure nodes)
    return float(entropy) if entropy > 0 else 0.0
