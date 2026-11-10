import numpy as np

def lda_classify(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns one predicted label for each test row using Linear Discriminant Analysis.
    """
    # Convert feature matrices to NumPy float64 arrays
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.int64)
    X_test = np.array(X_test, dtype=np.float64)
    
    n, d = X_train.shape
    classes = np.unique(y_train)
    classes.sort()  # Ensure classes are sorted to handle ties correctly
    K = len(classes)
    
    means = {}
    priors = {}
    scatter = np.zeros((d, d), dtype=np.float64)
    
    # Estimate each class mean, prior, and accumulate centered scatter
    for c in classes:
        X_c = X_train[y_train == c]
        mu_c = np.mean(X_c, axis=0)
        means[c] = mu_c
        priors[c] = len(X_c) / n
        
        X_c_centered = X_c - mu_c
        scatter += X_c_centered.T @ X_c_centered
        
    # Form the pooled within-class covariance with regularization
    Sigma = scatter / (n - K) + 1e-6 * np.eye(d)
    
    # Precompute weights using np.linalg.solve to avoid explicit inversion
    v = {}
    constant_term = {}
    for c in classes:
        mu_c = means[c]
        v_c = np.linalg.solve(Sigma, mu_c)
        v[c] = v_c
        constant_term[c] = -0.5 * np.dot(mu_c, v_c) + np.log(priors[c])
        
    preds = []
    for x in X_test:
        best_score = -float('inf')
        best_c = None
        for c in classes:
            score = np.dot(x, v[c]) + constant_term[c]
            # Strict inequality ensures that the smallest label is chosen in case of a tie
            if best_c is None or score > best_score:
                best_score = score
                best_c = c
        preds.append(int(best_c))
        
    return preds
