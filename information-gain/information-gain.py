import numpy as np

def information_gain(y: list, split_mask: list) -> float:
    """
    Returns the information gain as a float.
    """
    y_arr = np.asarray(y)
    mask = np.asarray(split_mask, dtype=bool)
    
    n = len(y_arr)
    if n == 0:
        return 0.0
        
    y_left = y_arr[mask]
    y_right = y_arr[~mask]
    
    # Return 0.0 if either split child is empty
    if len(y_left) == 0 or len(y_right) == 0:
        return 0.0

    def entropy(labels: np.ndarray) -> float:
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return float(-np.sum(probs * np.log2(probs)))

    h_parent = entropy(y_arr)
    h_left = entropy(y_left)
    h_right = entropy(y_right)
    
    n_l = len(y_left)
    n_r = len(y_right)
    
    ig = h_parent - (n_l / n) * h_left - (n_r / n) * h_right
    return float(ig)
