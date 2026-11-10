import numpy as np
from scipy.stats import chi2

def chi_square_test(observed: list, expected: list) -> list:
    """
    Returns the chi-square statistic, degrees of freedom, p-value, and decision.
    """
    # Convert inputs to numpy arrays for vectorized calculations
    obs = np.array(observed, dtype=float)
    exp = np.array(expected, dtype=float)
    
    # Calculate the Chi-Square statistic
    chi2_stat = np.sum(((obs - exp) ** 2) / exp)
    
    # Degrees of freedom (k - 1)
    df = len(observed) - 1
    
    # Upper-tail p-value using the Survival Function (sf = 1 - cdf)
    p_value = chi2.sf(chi2_stat, df)
    
    # Check if the p-value is below the 0.05 significance level
    is_significant = bool(p_value < 0.05)
    
    # Return results rounded to 4 decimal places where applicable
    return [round(chi2_stat, 4), df, round(p_value, 4), is_significant]
