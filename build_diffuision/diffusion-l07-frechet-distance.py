import numpy as np

def frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """
    Returns the squared Frechet distance between Gaussian fits of the two feature sets as a Python float.
    """
    a = np.asarray(features_a, dtype=np.float64)
    b = np.asarray(features_b, dtype=np.float64)

    # 1. Empirical means (shape: (D,))
    mu_a = np.mean(a, axis=0)
    mu_b = np.mean(b, axis=0)

    # 2. Unbiased sample covariances (shape: (D, D), divisor N - 1)
    # Using atleast_2d ensures D=1 returns a (1, 1) matrix instead of a scalar
    sigma_a = np.atleast_2d(np.cov(a, rowvar=False, ddof=1))
    sigma_b = np.atleast_2d(np.cov(b, rowvar=False, ddof=1))

    # 3. Symmetric square root of sigma_a via eigendecomposition
    eigvals_a, eigvecs_a = np.linalg.eigh(sigma_a)
    eigvals_a = np.clip(eigvals_a, 0.0, None)
    sigma_a_sqrt = (eigvecs_a * np.sqrt(eigvals_a)) @ eigvecs_a.T

    # 4. Eigenvalues of symmetric product M = sigma_a^(1/2) @ sigma_b @ sigma_a^(1/2)
    sym_prod = sigma_a_sqrt @ sigma_b @ sigma_a_sqrt
    sym_prod = 0.5 * (sym_prod + sym_prod.T)  # Enforce exact symmetry
    eigvals_prod = np.linalg.eigvalsh(sym_prod)
    eigvals_prod = np.clip(eigvals_prod, 0.0, None)

    # 5. Assemble squared Frechet distance
    diff = mu_a - mu_b
    mean_term = np.dot(diff, diff)
    trace_term = np.trace(sigma_a) + np.trace(sigma_b) - 2.0 * np.sum(np.sqrt(eigvals_prod))

    return float(mean_term + trace_term)
