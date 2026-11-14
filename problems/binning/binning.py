import math

def binning(values: list, num_bins: int) -> list:
    """
    Returns the equal-width bin index of every value.
    """
    min_val = min(values)
    max_val = max(values)
    
    # Edge case: If all values are identical, assign all to bin 0
    if min_val == max_val:
        return [0] * len(values)
    
    # Calculate equal bin width
    width = (max_val - min_val) / num_bins
    
    # Compute 0-indexed bin for each value and clamp the max value to num_bins - 1
    result = [
        min(math.floor((x - min_val) / width), num_bins - 1)
        for x in values
    ]
    
    return result
