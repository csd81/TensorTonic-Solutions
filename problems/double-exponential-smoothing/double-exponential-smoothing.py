def double_exponential_smoothing(series: list, alpha: float, beta: float) -> list:
    """
    Returns the smoothed level at every time step.
    """
    # Initialize level with the first observation and trend with the first difference
    level = series[0]
    trend = series[1] - series[0]
    
    # Store all level values, starting with the initial level
    levels = [level]
    
    # Iterate through the remaining observations starting from t = 1
    for t in range(1, len(series)):
        prev_level = level
        
        # Update level first
        level = alpha * series[t] + (1 - alpha) * (prev_level + trend)
        
        # Use the newly updated level to update trend
        trend = beta * (level - prev_level) + (1 - beta) * trend
        
        levels.append(level)
        
    return levels
