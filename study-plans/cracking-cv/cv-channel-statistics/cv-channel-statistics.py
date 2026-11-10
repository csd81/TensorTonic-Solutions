import numpy as np

def channel_statistics(batch: list) -> dict:
    """
    Returns mean and std lists of C floats in a dictionary, rounded to four decimals.
    """
    # Convert input list to a numpy array for efficient multi-dimensional operations
    arr = np.array(batch, dtype=np.float64)
    
    # Calculate mean and population standard deviation (ddof=0) 
    # across all axes except the last one (channels)
    means = np.mean(arr, axis=(0, 1, 2))
    stds = np.std(arr, axis=(0, 1, 2))
    
    # Round results to 4 decimal places and convert back to native Python floats
    mean_list = [round(float(m), 4) for m in means]
    std_list = [round(float(s), 4) for s in stds]
    
    return {
        "mean": mean_list,
        "std": std_list
    }
