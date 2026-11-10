def precision_recall_ap(y_true: list, y_scores: list) -> tuple:
    """
    Returns precision values, recall values, and average precision, 
    rounded to 4 decimal places.
    """
    # Total positive labels in the dataset
    total_positives = sum(y_true)
    
    # Pair indices, labels, and scores together to perform a stable sort by score descending
    # Python's built-in sorted() is stable by default.
    paired = sorted(enumerate(zip(y_true, y_scores)), key=lambda x: x[1][1], reverse=True)
    
    # Initialize the curve with the specified start points
    precision_list = [1.0]
    recall_list = [0.0]
    
    current_tp = 0
    current_fp = 0
    
    i = 0
    n = len(paired)
    
    # Process rows grouped by identical scores
    while i < n:
        current_score = paired[i][1][1]
        
        # Aggregate all elements belonging to the same score threshold group
        while i < n and paired[i][1][1] == current_score:
            label = paired[i][1][0]
            if label == 1:
                current_tp += 1
            else:
                current_fp += 1
            i += 1
            
        # Calculate precision and recall for the current threshold group
        precision = current_tp / (current_tp + current_fp)
        recall = current_tp / total_positives
        
        precision_list.append(precision)
        recall_list.append(recall)
        
    # Compute step-function Average Precision (AP)
    ap = 0.0
    for idx in range(1, len(recall_list)):
        delta_recall = recall_list[idx] - recall_list[idx - 1]
        ap += delta_recall * precision_list[idx]
        
    # Round all values to 4 decimal places
    precision_rounded = [round(p, 4) for p in precision_list]
    recall_rounded = [round(r, 4) for r in recall_list]
    ap_rounded = round(ap, 4)
    
    return precision_rounded, recall_rounded, ap_rounded
