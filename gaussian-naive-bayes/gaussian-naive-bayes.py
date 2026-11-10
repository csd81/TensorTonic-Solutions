import math

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns a predicted class label for every test sample.
    """
    n = len(X_train)
    n_features = len(X_train[0])
    classes = list(set(y_train))
    eps = 1e-9

    # Group training samples by class
    class_samples = {c: [] for c in classes}
    for x, y in zip(X_train, y_train):
        class_samples[y].append(x)

    # Compute priors, means, and variances for each class
    priors = {}
    means = {}
    variances = {}

    for c in classes:
        samples = class_samples[c]
        n_c = len(samples)
        priors[c] = n_c / n
        
        # Initialize mean and variance tracking for each feature
        means[c] = [0.0] * n_features
        variances[c] = [0.0] * n_features
        
        # Calculate feature means for class c
        for j in range(n_features):
            means[c][j] = sum(sample[j] for sample in samples) / n_c
            
        # Calculate feature population variances for class c (+ epsilon)
        for j in range(n_features):
            sq_diff_sum = sum((sample[j] - means[c][j]) ** 2 for sample in samples)
            variances[c][j] = (sq_diff_sum / n_c) + eps

    # Predict for each test sample
    predictions = []
    for x in X_test:
        best_class = None
        max_log_posterior = -float('inf')
        
        for c in classes:
            # Start with log P(c)
            log_posterior = math.log(priors[c])
            
            # Add log likelihood for each feature
            for j in range(n_features):
                var = variances[c][j]
                mean = means[c][j]
                val = x[j]
                
                # Formula element: -0.5 * log(2 * pi * variance) - ((x - mean)^2 / (2 * variance))
                log_likelihood = -0.5 * math.log(2 * math.pi * var) - ((val - mean) ** 2) / (2 * var)
                log_posterior += log_likelihood
                
            if log_posterior > max_log_posterior:
                max_log_posterior = log_posterior
                best_class = c
                
        predictions.append(best_class)

    return predictions
