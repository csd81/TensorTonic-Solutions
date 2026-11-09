def token_classification(logits: list, label_names: list, gold_labels: list) -> dict:
    """
    Returns a dictionary with predictions (label strings) and accuracy (float).
    """
    predictions = []
    correct_count = 0
    
    for i, token_logits in enumerate(logits):
        # Find the max logit value
        max_val = max(token_logits)
        
        # Break ties by selecting the smallest index (first occurrence)
        pred_idx = token_logits.index(max_val)
        
        # Map index to the label name
        pred_label = label_names[pred_idx]
        predictions.append(pred_label)
        
        # Check against the gold label
        if pred_label == gold_labels[i]:
            correct_count += 1
            
    # Calculate accuracy and round to 4 decimal places
    accuracy = round(correct_count / len(logits), 4)
    
    return {
        "predictions": predictions,
        "accuracy": accuracy
    }
