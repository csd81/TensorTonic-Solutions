def expected_calibration_error(y_true: list, y_pred: list, n_bins: int) -> float:
    """
    Returns the calibration error as a float.
    """
    n = len(y_true)
    bin_sums_true = [0.0] * n_bins
    bin_sums_pred = [0.0] * n_bins
    bin_counts = [0] * n_bins

    for yt, yp in zip(y_true, y_pred):
        # Assign probability 1.0 to the final bin; otherwise scale by n_bins
        bin_idx = min(int(yp * n_bins), n_bins - 1)
        
        bin_sums_true[bin_idx] += yt
        bin_sums_pred[bin_idx] += yp
        bin_counts[bin_idx] += 1

    ece = 0.0
    for m in range(n_bins):
        count = bin_counts[m]
        if count > 0:
            acc = bin_sums_true[m] / count
            conf = bin_sums_pred[m] / count
            ece += (count / n) * abs(acc - conf)

    return float(ece)
