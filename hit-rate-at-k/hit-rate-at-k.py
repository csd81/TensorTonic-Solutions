def hit_rate_at_k(recommendations: list, ground_truth: list, k: int) -> float:
    """
    Returns the fraction of users with a relevant item in their first k recommendations.
    """
    if not recommendations:
        return 0.0

    hits = 0
    for recs, truth in zip(recommendations, ground_truth):
        top_k = recs[:k]
        truth_set = set(truth)
        if any(item in truth_set for item in top_k):
            hits += 1

    return hits / len(recommendations)
