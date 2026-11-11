import numpy as np


def chi2_independence(C: list) -> dict:
    """
    Returns a dictionary with chi2 and expected.
    """
    observed = np.asarray(C, dtype=np.float64)

    # Compute row totals, column totals, and grand total
    row_totals = np.sum(observed, axis=1)
    col_totals = np.sum(observed, axis=0)
    grand_total = np.sum(observed)

    # Build the expected count table using an outer product
    expected = np.outer(row_totals, col_totals) / grand_total

    # Compute the chi-square statistic
    chi2 = float(np.sum(((observed - expected) ** 2) / expected))

    return {"chi2": chi2, "expected": expected}
