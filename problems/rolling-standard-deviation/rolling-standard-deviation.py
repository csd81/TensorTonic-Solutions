import math

def rolling_std(values: list, window_size: int) -> list:
    """
    Returns the population standard deviation of every complete window.
    """
    result = []
    num_windows = len(values) - window_size + 1
    
    for i in range(num_windows):
        window = values[i : i + window_size]
        mean = sum(window) / window_size
        variance = sum((x - mean) ** 2 for x in window) / window_size
        result.append(math.sqrt(variance))
        
    return result
