import math

def adjusted_cosine_similarity(ratings_matrix: list, item_i: int, item_j: int) -> float:
    """
    Returns the adjusted cosine similarity between the requested items.
    """
    numerator = 0.0
    sum_sq_i = 0.0
    sum_sq_j = 0.0

    for user_ratings in ratings_matrix:
        rating_i = user_ratings[item_i]
        rating_j = user_ratings[item_j]

        # Include only users who rated both requested items
        if rating_i == 0 or rating_j == 0:
            continue

        # Compute user mean from all nonzero ratings
        nonzero_ratings = [r for r in user_ratings if r != 0]
        user_mean = sum(nonzero_ratings) / len(nonzero_ratings)

        # Center both ratings around the user's mean
        diff_i = rating_i - user_mean
        diff_j = rating_j - user_mean

        numerator += diff_i * diff_j
        sum_sq_i += diff_i ** 2
        sum_sq_j += diff_j ** 2

    denominator = math.sqrt(sum_sq_i) * math.sqrt(sum_sq_j)

    if denominator == 0.0:
        return 0.0

    return float(numerator / denominator)
