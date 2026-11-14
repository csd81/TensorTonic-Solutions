import numpy as np

def majority_classifier(y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a one-dimensional NumPy array.
    """
    y = np.asarray(y_train, dtype=int)
    
    # Get unique values, their first-appearance indices, and their frequencies
    values, first_indices, counts = np.unique(y, return_index=True, return_counts=True)
    
    # Sort by highest count first, then by earliest appearance (lowest index) to break ties
    best_idx = np.lexsort((first_indices, -counts))[0]
    majority_label = values[best_idx]
    
    # Return an integer array matching the number of samples in X_test
    return np.full(len(X_test), majority_label, dtype=int)
