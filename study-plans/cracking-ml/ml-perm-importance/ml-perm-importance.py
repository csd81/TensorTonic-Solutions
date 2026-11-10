import numpy as np

def permutation_importance(X: list, y: list, predict_fn, n_repeats: int = 5, seed: int = 42) -> list:
    """
    Returns one permutation importance per feature.
    """
    # 1. Convert inputs to appropriate NumPy arrays
    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y)
    
    n_samples, n_features = X_arr.shape
    
    # 2. Define a helper function to calculate accuracy
    def get_accuracy(X_data, y_true):
        preds = predict_fn(X_data)
        return np.mean(preds == y_true)
        
    # 3. Compute baseline accuracy once before any permutation
    baseline_acc = get_accuracy(X_arr, y_arr)
    
    # 4. Initialize one NumPy RandomState to be used sequentially
    rng = np.random.RandomState(seed)
    
    importances = []
    
    # 5. Process features from left to right (j from 0 to n_features - 1)
    for j in range(n_features):
        total_drop = 0.0
        
        for r in range(n_repeats):
            # Copy X before replacing one column
            X_permuted = X_arr.copy()
            
            # Shuffle the j-th feature column
            X_permuted[:, j] = rng.permutation(X_permuted[:, j])
            
            # Compute accuracy after permutation
            permuted_acc = get_accuracy(X_permuted, y_arr)
            
            # Accumulate the drop in accuracy
            total_drop += (baseline_acc - permuted_acc)
            
        # Return the mean accuracy drop for each feature
        mean_drop = total_drop / n_repeats
        importances.append(round(mean_drop, 4))
        
    return importances
