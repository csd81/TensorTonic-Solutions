def popularity_ranking(items: list, min_votes: int, global_mean: float) -> list:
    """
    Returns the weighted rating for every item.
    """
    return [
        (v / (v + min_votes)) * r + (min_votes / (v + min_votes)) * global_mean
        for r, v in items
    ]
