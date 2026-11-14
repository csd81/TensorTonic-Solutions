def simple_moving_average(values: list, window_size: int) -> list:
    """
    Returns the mean of every complete sliding window.
    """
    window_sum = sum(values[:window_size])
    result = [window_sum / window_size]

    for i in range(window_size, len(values)):
        window_sum += values[i] - values[i - window_size]
        result.append(window_sum / window_size)

    return result
