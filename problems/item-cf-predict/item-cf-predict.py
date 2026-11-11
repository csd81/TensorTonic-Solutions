def item_cf_predict(user_ratings: list, item_similarities: list, target: int) -> float:
    """
    Returns the similarity-weighted rating prediction.
    """
    weighted_sum = 0.0
    sim_sum = 0.0

    for i, (rating, sim) in enumerate(zip(user_ratings, item_similarities)):
        if i == target:
            continue
        if rating > 0 and sim > 0:
            weighted_sum += sim * rating
            sim_sum += sim

    if sim_sum == 0.0:
        return 0.0

    return float(weighted_sum / sim_sum)
