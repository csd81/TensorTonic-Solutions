def baseline_predict(ratings_matrix: list, target_pairs: list) -> list:
    """
    Returns the baseline predictions for the requested user-item pairs.
    """
    n_users = len(ratings_matrix)
    n_items = len(ratings_matrix[0]) if n_users > 0 else 0

    user_sums = [0.0] * n_users
    user_counts = [0] * n_users
    item_sums = [0.0] * n_items
    item_counts = [0] * n_items

    total_sum = 0.0
    total_count = 0

    # 1. Accumulate nonzero rating sums and counts in one pass
    for u in range(n_users):
        for i in range(n_items):
            rating = ratings_matrix[u][i]
            if rating != 0:
                user_sums[u] += rating
                user_counts[u] += 1
                item_sums[i] += rating
                item_counts[i] += 1
                total_sum += rating
                total_count += 1

    # 2. Compute global mean
    mu = total_sum / total_count

    # 3. Compute user and item biases (0.0 if no observed ratings)
    user_biases = [
        (user_sums[u] / user_counts[u]) - mu if user_counts[u] > 0 else 0.0
        for u in range(n_users)
    ]
    item_biases = [
        (item_sums[i] / item_counts[i]) - mu if item_counts[i] > 0 else 0.0
        for i in range(n_items)
    ]

    # 4. Generate predictions in target_pairs order
    predictions = [
        float(mu + user_biases[u] + item_biases[i]) for u, i in target_pairs
    ]

    return predictions
