def jaccard_similarity(set_a: list, set_b: list) -> float:
    """
    Returns the Jaccard similarity of the two item collections.
    """
    a = set(set_a)
    b = set(set_b)
    
    union_len = len(a | b)
    if union_len == 0:
        return 0.0
        
    intersection_len = len(a & b)
    return intersection_len / union_len
