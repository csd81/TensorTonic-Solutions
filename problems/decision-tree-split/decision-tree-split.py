from collections import Counter


def decision_tree_split(X: list, y: list) -> list:
    """
    Returns the best feature index and threshold.
    """
    n_samples = len(X)
    n_features = len(X[0])

    def gini_impurity(labels: list) -> float:
        total = len(labels)
        if total == 0:
            return 0.0
        counts = Counter(labels)
        return 1.0 - sum((count / total) ** 2 for count in counts.values())

    parent_gini = gini_impurity(y)
    best_gain = -1.0
    best_feature = 0
    best_threshold = 0.0

    for feat_idx in range(n_features):
        unique_vals = sorted(set(row[feat_idx] for row in X))
        if len(unique_vals) < 2:
            continue

        for i in range(len(unique_vals) - 1):
            threshold = (unique_vals[i] + unique_vals[i + 1]) / 2.0

            left_y = []
            right_y = []
            for row, label in zip(X, y):
                if row[feat_idx] <= threshold:
                    left_y.append(label)
                else:
                    right_y.append(label)

            weighted_gini = (len(left_y) / n_samples) * gini_impurity(left_y) + (
                len(right_y) / n_samples
            ) * gini_impurity(right_y)
            gain = parent_gini - weighted_gini

            # Strict '>' preserves smallest feature index first, then smallest threshold on ties
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_feature = feat_idx
                best_threshold = float(threshold)

    return [best_feature, best_threshold]
