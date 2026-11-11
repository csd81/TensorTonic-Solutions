import math

def compute_monitoring_metrics(system_type: str, y_true: list, y_pred: list) -> dict:
    """
    Returns a dictionary of metrics.
    """
    if system_type == "classification":
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        n = len(y_true)
        accuracy = (tp + tn) / n if n > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    if system_type == "regression":
        n = len(y_true)
        mae = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n
        mse = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n
        rmse = math.sqrt(mse)

        return {
            "mae": float(mae),
            "rmse": float(rmse),
        }

    if system_type == "ranking":
        # Sort indices by descending predicted score; ascending index preserves input order on ties
        ranked_indices = sorted(range(len(y_pred)), key=lambda i: (-y_pred[i], i))
        top_3 = ranked_indices[:3]

        hits_at_3 = sum(1 for i in top_3 if y_true[i] == 1)
        total_relevant = sum(1 for yt in y_true if yt == 1)

        precision_at_3 = hits_at_3 / 3.0
        recall_at_3 = hits_at_3 / total_relevant if total_relevant > 0 else 0.0

        return {
            "precision_at_3": float(precision_at_3),
            "recall_at_3": float(recall_at_3),
        }

    raise ValueError(f"Unsupported system_type: {system_type}")
