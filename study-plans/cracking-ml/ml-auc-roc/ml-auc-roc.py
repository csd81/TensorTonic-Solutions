import numpy as np

def auc_roc(y_true: list, y_scores: list) -> tuple:
    """
    Returns false-positive rates, true-positive rates, and AUC from scratch.
    """
    P = sum(1 for y in y_true if y == 1)
    N = sum(1 for y in y_true if y == 0)
    
    # Pair indices with data to ensure a stable sort by score descending
    indexed_data = list(enumerate(zip(y_scores, y_true)))
    indexed_data.sort(key=lambda x: (-x[1][0], x[0]))
    
    fpr_list = [0.0]
    tpr_list = [0.0]
    
    tp = 0
    fp = 0
    
    i = 0
    n = len(indexed_data)
    while i < n:
        current_score = indexed_data[i][1][0]
        # Process all elements with identical scores as a single group
        while i < n and indexed_data[i][1][0] == current_score:
            if indexed_data[i][1][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        fpr_list.append(fp / N)
        tpr_list.append(tp / P)
        
    # Calculate Area Under Curve (AUC) using the trapezoidal rule
    auc = 0.0
    for j in range(len(fpr_list) - 1):
        dfpr = fpr_list[j+1] - fpr_list[j]
        auc += dfpr * (tpr_list[j] + tpr_list[j+1]) / 2.0
            
    # Round all values to 4 decimal places as requested
    fpr_rounded = [round(x, 4) for x in fpr_list]
    tpr_rounded = [round(x, 4) for x in tpr_list]
    auc_rounded = round(auc, 4)
    
    return fpr_rounded, tpr_rounded, auc_rounded
