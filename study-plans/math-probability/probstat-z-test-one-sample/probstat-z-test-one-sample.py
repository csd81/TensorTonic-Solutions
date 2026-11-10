from math import sqrt
from scipy.stats import norm

def z_test_one_sample(x_bar: float, mu_0: float, sigma: float, n: int, alpha: float) -> list:
    """
    Returns the z-statistic, two-sided p-value, and rejection decision.
    """
    # Compute the standard error
    standard_error = sigma / sqrt(n)
    
    # Compute the signed z-statistic
    z = (x_bar - mu_0) / standard_error
    
    # Compute the two-sided p-value
    p_value = 2 * norm.cdf(-abs(z))
    
    # Compare the unrounded p-value with alpha
    reject = p_value < alpha
    
    # Return the rounded results and decision
    return [round(z, 4), round(p_value, 4), reject]
