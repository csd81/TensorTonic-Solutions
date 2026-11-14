import numpy as np


def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    average: str = "micro",
    pos_label: int = 1,
) -> dict:
    """Returns a dictionary containing accuracy, precision, recall, and f1 rounded to six decimals."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    # Overall accuracy across all samples
    accuracy = float(np.mean(y_t == y_p))

    def safe_div(num: float, den: float) -> float:
        return float(num / den) if den != 0 else 0.0

    # Determine the set of classes to evaluate
    classes = np.unique(np.concatenate([y_t, y_p]))

    if average == "binary":
        tp = np.sum((y_t == pos_label) & (y_p == pos_label))
        fp = np.sum((y_t != pos_label) & (y_p == pos_label))
        fn = np.sum((y_t == pos_label) & (y_p != pos_label))

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

    elif average == "micro":
        # Sum TP, FP, and FN across all classes before computing metrics
        total_tp = sum(np.sum((y_t == c) & (y_p == c)) for c in classes)
        total_fp = sum(np.sum((y_t != c) & (y_p == c)) for c in classes)
        total_fn = sum(np.sum((y_t == c) & (y_p != c)) for c in classes)

        precision = safe_div(total_tp, total_tp + total_fp)
        recall = safe_div(total_tp, total_tp + total_fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

    elif average in ("macro", "weighted"):
        precisions, recalls, f1s, supports = [], [], [], []

        for c in classes:
            tp = np.sum((y_t == c) & (y_p == c))
            fp = np.sum((y_t != c) & (y_p == c))
            fn = np.sum((y_t == c) & (y_p != c))
            support = np.sum(y_t == c)

            p = safe_div(tp, tp + fp)
            r = safe_div(tp, tp + fn)
            f = safe_div(2 * p * r, p + r)

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
            supports.append(support)

        if average == "macro":
            precision = float(np.mean(precisions))
            recall = float(np.mean(recalls))
            f1 = float(np.mean(f1s))
        else:  # weighted
            total_support = np.sum(supports)
            precision = safe_div(np.sum(np.array(precisions) * supports), total_support)
            recall = safe_div(np.sum(np.array(recalls) * supports), total_support)
            f1 = safe_div(np.sum(np.array(f1s) * supports), total_support)

    else:
        raise ValueError(f"Unsupported average type: {average}")

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }
