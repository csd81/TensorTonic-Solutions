import math

def novelty_score(recommendations: list, item_counts: list, n_users: int) -> float:
    """
    Returns the average self-information of the recommended items.
    """
    if not recommendations:
        return 0.0
    
    total_novelty = sum(-math.log2(item_counts[i] / n_users) for i in recommendations)
    return total_novelty / len(recommendations)
