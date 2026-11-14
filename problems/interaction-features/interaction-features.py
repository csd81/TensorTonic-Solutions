def interaction_features(X: list) -> list:
    """
    Returns original features followed by unique pairwise products.
    """
    result = []
    for row in X:
        d = len(row)
        interactions = [row[i] * row[j] for i in range(d) for j in range(i + 1, d)]
        result.append(list(row) + interactions)
    return result
