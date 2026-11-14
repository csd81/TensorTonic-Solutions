import math

def compute_monitoring_metrics(system_type: str, y_true: list, y_pred: list) -> dict:
    """
    Returns a dictionary of metrics.
    """
    n = len(y_true)
    
    if system_type == "classification":
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        
        accuracy = (tp + tn) / n
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        
    elif system_type == "regression":
        abs_errors = [abs(yt - yp) for yt, yp in zip(y_true, y_pred)]
        sq_errors = [(yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)]
        
        mae = sum(abs_errors) / n
        rmse = math.sqrt(sum(sq_errors) / n)
        
        return {
            "mae": mae,
            "rmse": rmse,
        }
        
    elif system_type == "ranking":
        # Sort by descending y_pred; Python's sort is stable, so ties retain input order
        paired = sorted(zip(y_pred, y_true), key=lambda x: x[0], reverse=True)
        
        top_3_relevant = sum(yt for _, yt in paired[:3])
        total_relevant = sum(y_true)
        
        precision_at_3 = top_3_relevant / 3.0
        recall_at_3 = top_3_relevant / total_relevant if total_relevant > 0 else 0.0
        
        return {
            "precision_at_3": precision_at_3,
            "recall_at_3": recall_at_3,
        }
