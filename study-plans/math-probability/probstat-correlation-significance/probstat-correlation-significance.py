import numpy as np
from scipy.stats import t

def correlation_analysis(x: list, y: list) -> dict:
    """
    Returns correlation, explained proportion, test statistic, p-value, and significance.
    """

    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n= len(x)
    
    r =  float(np.corrcoef(x,y)[0,1])
    r_squared = r**2
    
    t_stat = r * np.sqrt(n-2) / np.sqrt(1-r_squared)
    
    p_value = 2 * (1 - t.cdf(abs(t_stat), df = n-2))
    significant = bool(p_value<0.05)
    
    return {
        "r": round(float(r), 4),
        "r_squared": round(float(r_squared), 4),
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "significant": significant
    }