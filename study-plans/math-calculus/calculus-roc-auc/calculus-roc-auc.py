def auc_roc_trapezoidal(y_true: list, y_scores: list) -> dict:
    """
    Returns ROC coordinates (FPR, TPR) and the area under the curve (AUC)
    using the trapezoidal rule with stable tie-handling.
    """
    # Count total positive (P) and negative (N) examples
    P = sum(1 for y in y_true if y == 1)
    N = sum(1 for y in y_true if y == 0)
    
    # Python's built-in sorted() is guaranteed to be stable.
    # Sort indices by score in descending order.
    sorted_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
    
    # Initialize the rate lists starting at zero
    fpr = [0.0]
    tpr = [0.0]
    
    tp_count = 0
    fp_count = 0
    
    # Append one ROC point after processing each label
    for idx in sorted_indices:
        if y_true[idx] == 1:
            tp_count += 1
        else:
            fp_count += 1
        
        fpr.append(fp_count / N)
        tpr.append(tp_count / P)
        
    # Calculate Area Under the Curve using the Trapezoidal Rule
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2.0
        
    return {
        "fpr": fpr,
        "tpr": tpr,
        "auc": float(auc)
    }
