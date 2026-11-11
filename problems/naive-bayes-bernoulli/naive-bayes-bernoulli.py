import numpy as np

def naive_bayes_bernoulli(X_train: list, y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a NumPy array of log posteriors.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float64)

    classes = np.sort(np.unique(y_train))
    n_train = X_train.shape[0]
    n_features = X_train.shape[1]
    n_classes = len(classes)

    log_priors = np.zeros(n_classes, dtype=np.float64)
    theta = np.zeros((n_classes, n_features), dtype=np.float64)

    for idx, c in enumerate(classes):
        X_c = X_train[y_train == c]
        n_c = X_c.shape[0]
        log_priors[idx] = np.log(n_c / n_train)
        # Laplace smoothing: (N_jc + 1) / (N_c + 2)
        theta[idx] = (X_c.sum(axis=0) + 1.0) / (n_c + 2.0)

    # Vectorized log posterior:
    # log P(c) + X_test @ log(theta.T) + (1 - X_test) @ log(1 - theta.T)
    log_theta = np.log(theta)
    log_one_minus_theta = np.log(1.0 - theta)

    log_posteriors = (
        log_priors
        + X_test @ log_theta.T
        + (1.0 - X_test) @ log_one_minus_theta.T
    )

    return np.round(log_posteriors, 4)
