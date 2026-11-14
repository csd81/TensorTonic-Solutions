import numpy as np

def time_based_split(timestamps: list, train_end: int, gap: int, val_end: int) -> dict:
    """
    Returns a dictionary with train_idx and val_idx as sorted integer NumPy arrays.
    """
    ts = np.asarray(timestamps, dtype=np.int64)
    
    train_mask = ts < train_end
    val_mask = (ts >= train_end + gap) & (ts < val_end)
    
    return {
        "train_idx": np.flatnonzero(train_mask),
        "val_idx": np.flatnonzero(val_mask),
    }
