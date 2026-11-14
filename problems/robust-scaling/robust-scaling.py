def robust_scaling(values: list) -> list:
    """
    Returns values centered by the median and scaled by the interquartile range.
    """
    def get_median(arr: list) -> float:
        m = len(arr)
        mid = m // 2
        if m % 2 == 1:
            return float(arr[mid])
        return (arr[mid - 1] + arr[mid]) / 2.0

    n = len(values)
    if n == 1:
        return [0.0]

    sorted_vals = sorted(values)
    q2 = get_median(sorted_vals)

    mid = n // 2
    if n % 2 == 1:
        lower_half = sorted_vals[:mid]
        upper_half = sorted_vals[mid + 1:]
    else:
        lower_half = sorted_vals[:mid]
        upper_half = sorted_vals[mid:]

    q1 = get_median(lower_half)
    q3 = get_median(upper_half)
    iqr = q3 - q1

    if iqr == 0:
        return [float(x - q2) for x in values]

    return [(x - q2) / iqr for x in values]
