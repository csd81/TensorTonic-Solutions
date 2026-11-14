def moving_median(values: list, window_size: int) -> list:
    """
    Returns the median of every complete sliding window.
    """
    medians = []
    mid = window_size // 2
    is_even = (window_size % 2 == 0)

    for i in range(len(values) - window_size + 1):
        window = sorted(values[i : i + window_size])
        if is_even:
            median = (window[mid - 1] + window[mid]) / 2.0
        else:
            median = float(window[mid])
        medians.append(median)

    return medians
