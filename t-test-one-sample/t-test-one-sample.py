import numpy as np


def t_test_one_sample(x: list, mu0: float) -> float:
    """
    Returns the t-statistic as a float.
    """
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)

    mean_val = np.mean(arr)
    # Use ddof=1 for Bessel's correction (n - 1 denominator)
    std_val = np.std(arr, ddof=1)

    diff = mean_val - mu0

    # Handle zero sample variance edge cases
    if std_val == 0.0:
        if diff == 0.0:
            return 0.0
        return float(np.inf) if diff > 0.0 else float(-np.inf)

    standard_error = std_val / np.sqrt(n)
    t_stat = diff / standard_error

    return float(t_stat)
