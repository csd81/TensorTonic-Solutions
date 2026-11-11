import numpy as np

def summarize(data: list, axis: int) -> np.ndarray:
    """
    Returns float64 rows of mean, standard deviation, minimum, and maximum.
    """
    # Convert input list to a NumPy array with float64 precision
    arr = np.array(data, dtype=np.float64)
    
    # Calculate the required metrics along the given axis
    means = np.mean(arr, axis=axis)
    stds = np.std(arr, axis=axis)  # np.std defaults to population std (ddof=0)
    mins = np.min(arr, axis=axis)
    maxs = np.max(arr, axis=axis)
    
    # Stack them vertically into the final shape: (4, number of groups)
    return np.stack([means, stds, mins, maxs], axis=0)
