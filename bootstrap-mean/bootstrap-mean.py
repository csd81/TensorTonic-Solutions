import numpy as np


def bootstrap_mean(
    x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0
) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)

    # Initialize the seeded random number generator
    rng = np.random.default_rng(seed)

    # Draw (n_bootstrap, n) random indices with replacement and index into arr
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    resamples = arr[indices]

    # Compute the mean of each bootstrap resample
    boot_means = np.mean(resamples, axis=1)

    # Compute overall bootstrap mean and percentile interval bounds
    mean_val = float(np.mean(boot_means))
    alpha = (1.0 - ci) / 2.0
    lower, upper = np.quantile(boot_means, [alpha, 1.0 - alpha])

    return {
        "bootstrap_mean": mean_val,
        "lower": float(lower),
        "upper": float(upper),
    }
