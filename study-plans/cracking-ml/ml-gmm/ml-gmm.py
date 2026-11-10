import numpy as np

def gmm(X: list, k: int, max_iters: int = 100, seed: int = 42) -> tuple:
    """
    Fits a diagonal-covariance Gaussian mixture with expectation-maximization.
    Returns a tuple containing component labels, mean rows, and mixture weights.
    """
    X = np.array(X, dtype=np.float64)
    N, D = X.shape
    
    # Initialize component means from k distinct randomly selected rows
    np.random.seed(seed)
    initial_indices = np.random.choice(N, k, replace=False)
    means = X[initial_indices].copy()
    
    # Initialize all diagonal variances to one and weights to 1/k
    variances = np.ones((k, D), dtype=np.float64)
    weights = np.ones(k, dtype=np.float64) / k
    
    for _ in range(max_iters):
        # --- E-Step ---
        log_p = np.zeros((N, k))
        for j in range(k):
            log_pi = np.log(weights[j])
            term1 = np.log(2 * np.pi * variances[j])
            term2 = ((X - means[j]) ** 2) / variances[j]
            log_p[:, j] = log_pi - 0.5 * np.sum(term1 + term2, axis=1)
            
        # Normalize responsibilities with maximum-subtracted log values
        max_log_p = np.max(log_p, axis=1, keepdims=True)
        exp_log_p = np.exp(log_p - max_log_p)
        resp = exp_log_p / np.sum(exp_log_p, axis=1, keepdims=True)
        
        # --- M-Step ---
        N_j = np.sum(resp, axis=0)
        weights = N_j / N
        for j in range(k):
            if N_j[j] > 0:
                means[j] = np.sum(resp[:, j:j+1] * X, axis=0) / N_j[j]
                variances[j] = np.sum(resp[:, j:j+1] * (X - means[j])**2, axis=0) / N_j[j]
            # Add 1e-6 to every variance component
            variances[j] += 1e-6
            
    # Recompute responsibilities from the final parameters
    log_p = np.zeros((N, k))
    for j in range(k):
        log_pi = np.log(weights[j])
        term1 = np.log(2 * np.pi * variances[j])
        term2 = ((X - means[j]) ** 2) / variances[j]
        log_p[:, j] = log_pi - 0.5 * np.sum(term1 + term2, axis=1)
        
    max_log_p = np.max(log_p, axis=1, keepdims=True)
    exp_log_p = np.exp(log_p - max_log_p)
    resp = exp_log_p / np.sum(exp_log_p, axis=1, keepdims=True)
    
    # Compute component labels from final responsibilities
    labels = np.argmax(resp, axis=1).tolist()
    rounded_means = np.round(means, 4).tolist()
    rounded_weights = np.round(weights, 4).tolist()
    
    return labels, rounded_means, rounded_weights
