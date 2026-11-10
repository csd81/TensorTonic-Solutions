from scipy.stats import norm, t

def p_value_analysis(test_stat: float, dist_type: str, df: int, tail: str) -> dict:
    """
    Returns the p-value and significance classifications.
    """
    # Select the distribution type
    if dist_type == "z":
        dist = norm
    else:
        dist = t(df=df)
        
    # Calculate the unrounded p-value based on the test type (tail)
    if tail == "two":
        p_val = 2 * dist.sf(abs(test_stat))
    elif tail == "left":
        p_val = dist.cdf(test_stat)
    elif tail == "right":
        p_val = dist.sf(test_stat)
        
    # Build results matching unrounded p-value to thresholds, then round the p-value
    return {
        "p_value": round(p_val, 4),
        "significant_at_01": p_val < 0.01,
        "significant_at_05": p_val < 0.05,
        "significant_at_10": p_val < 0.10
    }
