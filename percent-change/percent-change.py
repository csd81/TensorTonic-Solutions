def percent_change(series: list) -> list:
    """
    Returns the fractional change between consecutive values.
    """
    result = []
    for i in range(1, len(series)):
        prev_val = series[i - 1]
        if prev_val == 0:
            result.append(0.0)
        else:
            change = (series[i] - prev_val) / prev_val
            result.append(change)
    return result
