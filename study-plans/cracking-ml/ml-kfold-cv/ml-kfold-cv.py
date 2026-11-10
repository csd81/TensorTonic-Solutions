import numpy as np

def kfold_cv(X: list, y: list, model_fn, k: int = 5, seed: int = 42) -> tuple:
    """
    Returns fold accuracies and their mean.
    """
    # 1. Convert inputs to appropriate NumPy arrays
    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y)
    
    n = len(X_arr)
    
    # 2. Shuffle row indices in place using a RandomState instance
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    
    # Rearrange data according to shuffled indices
    X_shuffled = X_arr[indices]
    y_shuffled = y_arr[indices]
    
    # 3. Create folds using np.array_split
    X_folds = np.array_split(X_shuffled, k)
    y_folds = np.array_split(y_shuffled, k)
    
    fold_accuracies = []
    
    # 4. Evaluate each fold
    for i in range(k):
        # Separate the validation fold
        X_val = X_folds[i]
        y_val = y_folds[i]
        
        # Concatenate all other folds in ascending order to create training sets
        X_train_list = [X_folds[j] for j in range(k) if j != i]
        y_train_list = [y_folds[j] for j in range(k) if j != i]
        
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        
        # Train model function and generate predictions
        predict_fn = model_fn(X_train, y_train)
        predictions = predict_fn(X_val)
        
        # Calculate accuracy for the current fold
        # Match types for proper equality comparison
        correct = np.sum(np.array(predictions) == np.array(y_val))
        accuracy = correct / len(y_val)
        
        # Round the fold accuracy first as requested
        fold_accuracies.append(round(accuracy, 4))
        
    # 5. Compute the mean accuracy and round it to 4 decimal places
    mean_accuracy = round(np.mean(fold_accuracies), 4)
    
    return fold_accuracies, mean_accuracy
