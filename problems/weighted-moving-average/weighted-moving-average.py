def weighted_moving_average(values: list, weights: list) -> list:
    """
    Returns the weighted average of every complete window.
    """
    k = len(weights)
    weight_sum = sum(weights)
    result = []

    for i in range(len(values) - k + 1):
        weighted_sum = sum(w * x for w, x in zip(weights, values[i : i + k]))
        result.append(weighted_sum / weight_sum)

    return result
