from collections import Counter


def cohens_kappa(rater1: list, rater2: list) -> float:
    """
    Returns Cohen's kappa as a float.
    """
    n = len(rater1)

    # 1. Observed agreement (p_o)
    matches = sum(1 for a, b in zip(rater1, rater2) if a == b)
    p_o = matches / n

    # 2. Expected agreement (p_e)
    counts1 = Counter(rater1)
    counts2 = Counter(rater2)
    labels = set(counts1.keys()) | set(counts2.keys())

    p_e = sum((counts1[k] / n) * (counts2[k] / n) for k in labels)

    # 3. Handle edge case and compute kappa
    if p_e == 1.0:
        return 1.0

    return float((p_o - p_e) / (1.0 - p_e))
