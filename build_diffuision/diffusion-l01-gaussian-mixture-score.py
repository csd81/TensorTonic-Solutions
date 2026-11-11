import numpy as np

def gaussian_mixture_score(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray
) -> np.ndarray:
    """
    Returns the score grad_x log p(x) of the isotropic Gaussian mixture at every row of x, as a float64 array of shape (N, D).
    """
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)

    N, D = x.shape

    # 1. Pairwise differences: (mu_k - x_n) -> shape (N, K, D)
    diff = means[None, :, :] - x[:, None, :]

    # 2. Squared Euclidean distances: ||x_n - mu_k||^2 -> shape (N, K)
    sq_dist = np.sum(diff ** 2, axis=-1)

    # 3. Unnormalized log responsibilities: log(pi_k) + log N(x_n; mu_k, sigma_k^2 I) -> shape (N, K)
    log_norm_const = -0.5 * D * np.log(2.0 * np.pi * variances[None, :])
    log_joint = np.log(weights[None, :]) + log_norm_const - 0.5 * sq_dist / variances[None, :]

    # 4. Log-sum-exp along components (axis=1) for numerical stability -> shape (N, K)
    max_log_joint = np.max(log_joint, axis=1, keepdims=True)
    log_sum_exp = max_log_joint + np.log(np.sum(np.exp(log_joint - max_log_joint), axis=1, keepdims=True))
    responsibilities = np.exp(log_joint - log_sum_exp)

    # 5. Component scores: (mu_k - x_n) / sigma_k^2 -> shape (N, K, D)
    component_scores = diff / variances[None, :, None]

    # 6. Weighted sum of component scores: sum_k r_k(x_n) * component_score_{n,k} -> shape (N, D)
    score = np.sum(responsibilities[:, :, None] * component_scores, axis=1)

    return score.astype(np.float64)
