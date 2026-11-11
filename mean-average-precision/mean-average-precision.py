import numpy as np


def mean_average_precision(
    y_true_list: list, y_score_list: list, k: int | None = None
) -> dict:
    """
    Returns a dictionary with map_value and ap_per_query.
    """
    ap_per_query = []

    for y_true, y_score in zip(y_true_list, y_score_list):
        rel = np.asarray(y_true, dtype=np.float64)
        scores = np.asarray(y_score, dtype=np.float64)

        total_relevant = float(np.sum(rel))
        if total_relevant == 0.0 or rel.size == 0:
            ap_per_query.append(0.0)
            continue

        # Sort items by descending score (stable sort preserves original order on ties)
        desc_idx = np.argsort(-scores, kind="mergesort")
        sorted_rel = rel[desc_idx]

        # Apply cutoff k if provided
        if k is not None:
            sorted_rel = sorted_rel[:k]

        if sorted_rel.size == 0:
            ap_per_query.append(0.0)
            continue

        # Precision at every rank r = 1, 2, ..., len(sorted_rel)
        ranks = np.arange(1, sorted_rel.size + 1, dtype=np.float64)
        precisions = np.cumsum(sorted_rel) / ranks

        # Sum precision only at relevant ranks and normalize by total relevant count R
        ap = float(np.sum(precisions * sorted_rel) / total_relevant)
        ap_per_query.append(round(ap, 6))

    map_value = round(float(np.mean(ap_per_query)), 6) if ap_per_query else 0.0

    return {
        "map_value": map_value,
        "ap_per_query": ap_per_query,
    }
