def top_k_recommendations(scores: list, rated_indices: list, k: int) -> list:
    """
    Returns the highest-scoring unrated item indices.
    """
    rated_set = set(rated_indices)
    
    # Keep only indices that have not been rated
    unrated_indices = [i for i in range(len(scores)) if i not in rated_set]
    
    # Sort by score descending (-scores[i]), then by index ascending (i)
    unrated_indices.sort(key=lambda i: (-scores[i], i))
    
    # Return up to the top k indices
    return unrated_indices[:k]
