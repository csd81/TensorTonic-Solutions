import numpy as np


def roc_curve(y_true: list, y_score: list) -> dict:
    """
    Returns a dictionary with fpr, tpr, and thresholds.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    # 1. Sort samples by descending score
    desc_idx = np.argsort(y_score)[::-1]
    y_true = y_true[desc_idx]
    y_score = y_score[desc_idx]

    # 2. Identify the last index of each unique score group
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # 3. Compute cumulative TP and FP at each unique score boundary
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = (threshold_idxs + 1) - tps

    # 4. Prepend the origin point (fpr=0, tpr=0, threshold=inf)
    tps = np.r_[0.0, tps]
    fps = np.r_[0.0, fps]
    thresholds = np.r_[np.inf, y_score[threshold_idxs]]

    # 5. Normalize by total positives (P) and negatives (N)
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }
