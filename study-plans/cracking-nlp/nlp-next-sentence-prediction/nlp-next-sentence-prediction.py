def next_sentence_prediction(sentence_a: list, sentence_b: list, threshold: float = 0.5) -> bool:
    """
    Returns a bool indicating whether the word-overlap threshold is met.
    """
    # Convert lists to sets to get unique tokens
    set_a = set(sentence_a)
    set_b = set(sentence_b)
    
    # 1. Handle the double-empty edge case
    if not set_a and not set_b:
        return True
        
    # 2. Compute Jaccard Similarity: |A ∩ B| / |A ∪ B|
    union_size = len(set_a | set_b)
    intersection_size = len(set_a & set_b)
    
    jaccard_similarity = intersection_size / union_size
    
    # 3. Return True if similarity is at least the threshold
    return jaccard_similarity >= threshold
