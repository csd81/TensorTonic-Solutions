import numpy as np
from scipy.stats import t

def t_test_one_sample(data: list, mu_0: float, alpha: float) -> dict:
    """
    Returns the t-statistic, degrees of freedom, p-value, and rejection decision.
    """
    n = len(data)
    df = n - 1
    
    # Calculate sample mean and sample standard deviation (ddof=1 for sample std)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    
    # Compute t-statistic
    t_stat = (sample_mean - mu_0) / (sample_std / np.sqrt(n))
    
    # Compute two-sided p-value
    # Use survival function sf (1 - cdf) of the absolute t-stat, multiplied by 2
    p_val = 2 * t.sf(np.abs(t_stat), df)
    
    # Compare unrounded p-value with alpha
    reject_null = bool(p_val < alpha)
    
    return {
        "degrees_of_freedom": int(df),
        "p_value": float(round(p_val, 4)),
        "reject_null": reject_null,
        "t_statistic": float(round(t_stat, 4))
    }
