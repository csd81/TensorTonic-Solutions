def choose_threshold(y_true: list, scores: list, min_precision: float) -> dict:
    """
    Returns the threshold with the highest recall whose precision is at least min_precision.
    """
    total_positives = sum(y_true)
    
    # Pair scores with labels and sort in descending order by score
    pairs = sorted(zip(scores, y_true), key=lambda x: x[0], reverse=True)
    
    tp = 0
    pred_pos = 0
    
    best_threshold = None
    best_precision = -1.0
    best_recall = -1.0
    
    n = len(pairs)
    i = 0
    
    while i < n:
        current_score = pairs[i][0]
        
        # Group all rows that share the exact same score
        while i < n and pairs[i][0] == current_score:
            pred_pos += 1
            if pairs[i][1] == 1:
                tp += 1
            i += 1
            
        precision = tp / pred_pos
        recall = tp / total_positives
        
        if precision >= min_precision:
            # Maximize recall first, break ties with higher precision
            if (recall > best_recall) or (recall == best_recall and precision > best_precision):
                best_recall = recall
                best_precision = precision
                best_threshold = float(current_score)
                
    if best_threshold is None:
        return {"threshold": None, "precision": None, "recall": None}
        
    return {
        "threshold": best_threshold,
        "precision": best_precision,
        "recall": best_recall,
    }
