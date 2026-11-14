def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the micro-averaged F1 score as a Python float rounded to four decimals.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    fp = len(y_true) - tp
    fn = fp  # Every wrong prediction adds 1 FP to the predicted class and 1 FN to the true class
    
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 0.0
        
    return round((2 * tp) / denominator, 4)
