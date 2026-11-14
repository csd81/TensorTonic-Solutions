def exponential_moving_average(values: list, alpha: float) -> list:
    """
    Returns the exponential moving average at every position.
    """
    if not values:
        return []
        
    ema = [values[0]]
    
    for x in values[1:]:
        current_ema = alpha * x + (1 - alpha) * ema[-1]
        ema.append(current_ema)
        
    return ema
