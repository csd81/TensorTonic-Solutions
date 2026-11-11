import numpy as np


def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    if strategy not in ("mean", "median"):
        raise ValueError(f"Unsupported strategy: {strategy}")

    # Create a fresh floating-point copy without mutating input X
    arr = np.array(X, dtype=np.float64, copy=True)
    nan_mask = np.isnan(arr)

    # Fast exit if there are no missing values
    if not np.any(nan_mask):
        return arr

    # Handle 1D input as a single feature
    if arr.ndim == 1:
        if np.all(nan_mask):
            fill_val = 0.0
        elif strategy == "mean":
            fill_val = np.nanmean(arr)
        else:
            fill_val = np.nanmedian(arr)
        arr[nan_mask] = fill_val
        return arr

    # Handle 2D input column by column
    all_nan_cols = np.all(nan_mask, axis=0)
    fill_vals = np.zeros(arr.shape[1], dtype=np.float64)
    valid_cols = ~all_nan_cols

    if np.any(valid_cols):
        if strategy == "mean":
            fill_vals[valid_cols] = np.nanmean(arr[:, valid_cols], axis=0)
        else:
            fill_vals[valid_cols] = np.nanmedian(arr[:, valid_cols], axis=0)

    # Replace NaNs using broadcasted column fill values
    arr = np.where(nan_mask, fill_vals, arr)
    return arr
