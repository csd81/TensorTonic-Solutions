from collections import Counter

def frequency_encoding(values: list) -> list:
    """
    Returns the relative frequency of every input value.
    """
    n = len(values)
    counts = Counter(values)
    return [counts[v] / n for v in values]
