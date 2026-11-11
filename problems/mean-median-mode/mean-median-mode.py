from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    counts = Counter(x)
    mode = min(counts, key=lambda value: (-counts[value],value))

    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "mode": float(mode)
    }