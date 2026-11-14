def autocorrelation(series: list, max_lag: int) -> list:
    """
    Returns normalized autocorrelation from lag zero through max_lag.
    """
    n = len(series)
    mean = sum(series) / n
    
    # Center the series around the mean
    centered = [x - mean for x in series]
    
    # Compute total variance (gamma_0)
    gamma_0 = sum(c * c for c in centered)
    
    # Handle constant series where variance is 0
    if gamma_0 == 0:
        return [1.0] + [0.0] * max_lag
    
    result = []
    for k in range(max_lag + 1):
        # Compute lagged covariance sum
        cov_k = sum(centered[t] * centered[t + k] for t in range(n - k))
        # Normalize and round to 6 decimal places
        r_k = round(cov_k / gamma_0, 6)
        result.append(r_k)
        
    return result
