def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    top_k = set(recommended[:k])
    relevant_set = set(relevant)
    
    hits = len(top_k & relevant_set)
    
    precision = hits / k
    recall = hits / len(relevant_set)
    
    return [float(precision), float(recall)]
