from math import sqrt
from scipy.stats import norm

def ab_test(control_visitors: int, control_conversions: int, treatment_visitors: int, treatment_conversions: int, alpha: float) -> list:
    """
    Returns both rates, the z-statistic, p-value, and significance decision.
    """
    # Calculate conversion rates
    p_c = control_conversions / control_visitors
    p_t = treatment_conversions / treatment_visitors
    
    # Calculate pooled conversion rate and standard error
    p_hat = (control_conversions + treatment_conversions) / (control_visitors + treatment_visitors)
    se = sqrt(p_hat * (1 - p_hat) * (1 / control_visitors + 1 / treatment_visitors))
    
    # Calculate z-statistic and two-sided p-value
    z = (p_t - p_c) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    # Make significance decision using the unrounded p-value
    is_significant = p_value < alpha
    
    return [
        round(p_c, 4),
        round(p_t, 4),
        round(z, 4),
        round(p_value, 4),
        is_significant
    ]
