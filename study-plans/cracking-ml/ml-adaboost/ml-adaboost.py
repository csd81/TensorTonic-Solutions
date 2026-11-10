import numpy as np

def adaboost_classify(X_train: list, y_train: list, X_test: list, n_estimators: int = 10, seed: int = 42) -> list:
    """
    Returns one boosted label per test row using an AdaBoost ensemble of decision stumps.
    """
    X = np.array(X_train, dtype=float)
    y = np.array(y_train, dtype=float)
    X_tst = np.array(X_test, dtype=float)
    
    n_samples, n_features = X.shape
    weights = np.ones(n_samples) / n_samples
    
    # Store ensemble components
    alphas = []
    stumps = [] # list of dicts: {'feature': j, 'threshold': t, 'polarity': p}
    
    for _ in range(n_estimators):
        min_error = float('inf')
        best_stump = None
        best_predictions = None
        
        # Search features, observed thresholds, and polarities in exactly that order
        for j in range(n_features):
            # Use unique observed values to ensure consistent threshold evaluation
            thresholds = np.unique(X[:, j])
            for t in thresholds:
                for polarity in [1, -1]:
                    # Make predictions for the current stump configuration
                    if polarity == 1:
                        preds = np.where(X[:, j] <= t, -1.0, 1.0)
                    else:
                        preds = np.where(X[:, j] <= t, 1.0, -1.0)
                    
                    # Calculate weighted classification error
                    error = np.sum(weights[y != preds])
                    
                    # Retain the very first stump that achieves the minimum error
                    if error < min_error:
                        min_error = error
                        best_stump = {'feature': j, 'threshold': t, 'polarity': polarity}
                        best_predictions = preds
        
        # Clip error to avoid log(0) or division by zero
        min_error = np.clip(min_error, 1e-12, 1.0 - 1e-12)
        
        # Calculate estimator weight (alpha)
        alpha = 0.5 * np.log((1.0 - min_error) / min_error)
        
        # Update sample weights
        weights *= np.exp(-alpha * y * best_predictions)
        weights /= np.sum(weights) # Normalize
        
        # Save to ensemble
        alphas.append(alpha)
        stumps.append(best_stump)
        
    # Aggregate predictions on the test set
    test_preds = np.zeros(len(X_tst))
    for alpha, stump in zip(alphas, stumps):
        j = stump['feature']
        t = stump['threshold']
        p = stump['polarity']
        
        if p == 1:
            preds = np.where(X_tst[:, j] <= t, -1.0, 1.0)
        else:
            preds = np.where(X_tst[:, j] <= t, 1.0, -1.0)
            
        test_preds += alpha * preds
        
    # Predict +1 when the final score is >= 0, and -1 otherwise
    final_labels = np.where(test_preds >= 0.0, 1, -1)
    return final_labels.tolist()
