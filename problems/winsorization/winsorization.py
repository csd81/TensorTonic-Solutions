import math


def winsorize(values: list, lower_pct: float, upper_pct: float) -> list:
    """
    Returns values clipped to the interpolated percentile bounds.
    """
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def get_percentile(pct: float) -> float:
        k = (n - 1) * pct / 100.0
        lo = math.floor(k)
        hi = math.ceil(k)
        weight = k - lo
        return sorted_vals[lo] + weight * (sorted_vals[hi] - sorted_vals[lo])

    lower_bound = get_percentile(lower_pct)
    upper_bound = get_percentile(upper_pct)

    return [
        lower_bound if v < lower_bound else (upper_bound if v > upper_bound else v)
        for v in values
    ]
