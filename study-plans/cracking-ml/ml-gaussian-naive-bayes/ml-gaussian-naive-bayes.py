import numpy as np

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns one predicted label for each test row.
    """
    # 1. Convert inputs to NumPy arrays
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=np.float64)
    
    # 2. Get sorted unique classes to ensure stable tie-breaking
    classes = np.unique(y_train)
    
    # 3. Precompute priors, means, and smoothed variances for each class
    class_priors = {}
    class_means = {}
    class_vars = {}
    
    n_samples = len(y_train)
    
    for c in classes:
        X_c = X_train[y_train == c]
        
        # Prior probability p(c)
        class_priors[c] = len(X_c) / n_samples
        
        # Feature means μ_cj
        class_means[c] = np.mean(X_c, axis=0)
        
        # Population variance (ddof=0) + smoothing epsilon (10^-9)
        class_vars[c] = np.var(X_c, axis=0, ddof=0) + 1e-9

    predictions = []
    
    # 4. Score each test query
    for x in X_test:
        best_score = -float('inf')
        best_class = None
        
        for c in classes:
            prior = class_priors[c]
            means = class_means[c]
            variances = class_vars[c]
            
            # Log prior: log p(c)
            log_prior = np.log(prior)
            
            # Summed feature likelihood: log p(x_j | c)
            # Formula element: log(2 * pi * sigma^2) + ((x - mu)^2 / sigma^2)
            log_likelihood_elements = np.log(2 * np.pi * variances) + ((x - means) ** 2) / variances
            log_likelihood = -0.5 * np.sum(log_likelihood_elements)
            
            # Total posterior score
            score = log_prior + log_likelihood
            
            # Select the class with the largest score. 
            # Because `classes` is sorted, '>' ensures that an exact tie 
            # keeps the smaller, previously encountered class label.
            if score > best_score:
                best_score = score
                best_class = c
                
        predictions.append(int(best_class))
        
    return predictions
