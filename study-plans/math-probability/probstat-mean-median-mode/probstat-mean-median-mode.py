import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns mean, median, and mode as Python floats in a dictionary.
    """
    mean_val = float(np.mean(x))
    median_val = float(np.median(x))

    candidates, counts = np.unique(x, return_counts=True)

    mode_val = float(candidates[np.argmax(counts)])

    return {
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val
    }
