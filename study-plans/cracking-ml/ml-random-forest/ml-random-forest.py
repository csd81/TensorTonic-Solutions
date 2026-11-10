import numpy as np

def gini_impurity(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)

class RandomTree:
    def __init__(self, max_depth: int, max_features_mode, rng: np.random.RandomState):
        self.max_depth = max_depth
        self.max_features_mode = max_features_mode
        self.rng = rng
        self.root = None

    def _get_m(self, d: int) -> int:
        if self.max_features_mode == "sqrt":
            return max(1, int(np.floor(np.sqrt(d))))
        elif self.max_features_mode == "log2":
            return max(1, int(np.floor(np.log2(d))))
        else:
            # Assumed to be a positive integer
            return min(d, int(self.max_features_mode))

    def _leaf_value(self, y: np.ndarray) -> int:
        classes, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)
        best_classes = classes[counts == max_count]
        return int(np.min(best_classes))

    def fit(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> dict:
        unique_y = np.unique(y)
        if len(unique_y) == 1:
            return {'type': 'leaf', 'value': int(unique_y[0])}
        
        if depth >= self.max_depth:
            return {'type': 'leaf', 'value': self._leaf_value(y)}

        n_samples, d = X.shape
        parent_gini = gini_impurity(y)
        best_gain = 0.0  # Must choose a strictly best positive Gini gain (> 0.0)
        best_split = None

        # Determine feature subset size m
        m = self._get_m(d)
        
        # Sample feature indices without replacement and sort them in ascending order
        sampled_features = self.rng.choice(d, size=m, replace=False)
        sampled_features = np.sort(sampled_features)

        for feature_idx in sampled_features:
            thresholds = np.sort(np.unique(X[:, feature_idx]))
            for thr in thresholds:
                left_mask = X[:, feature_idx] <= thr
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                gain = parent_gini - (len(y_left) / n_samples) * gini_impurity(y_left) - (len(y_right) / n_samples) * gini_impurity(y_right)

                # Choose the first strictly best positive Gini gain
                if gain > best_gain + 1e-11:
                    best_gain = gain
                    best_split = {
                        'feature': feature_idx,
                        'threshold': thr,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }

        if best_split is None:
            return {'type': 'leaf', 'value': self._leaf_value(y)}

        left_node = self.fit(X[best_split['left_mask']], y[best_split['left_mask']], depth + 1)
        right_node = self.fit(X[best_split['right_mask']], y[best_split['right_mask']], depth + 1)

        return {
            'type': 'split',
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': left_node,
            'right': right_node
        }

    def predict_row(self, node: dict, x: np.ndarray) -> int:
        if node['type'] == 'leaf':
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self.predict_row(node['left'], x)
        else:
            return self.predict_row(node['right'], x)

def random_forest_classify(X_train: list, y_train: list, X_test: list, n_estimators: int = 10, max_depth: int = 5, max_features = "sqrt", seed: int = 42) -> list:
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    n = len(X_train)
    
    # Shared single RandomState instance for both bootstrap and node feature sampling
    rng = np.random.RandomState(seed)
    all_tree_preds = []

    for _ in range(n_estimators):
        # Draw size-n bootstrap sample indices
        bootstrap_indices = rng.randint(0, n, size=n)
        X_b = X_train[bootstrap_indices]
        y_b = y_train[bootstrap_indices]

        # Initialize and build tree sharing the ensemble rng
        tree = RandomTree(max_depth=max_depth, max_features_mode=max_features, rng=rng)
        root_node = tree.fit(X_b, y_b)

        preds = [tree.predict_row(root_node, x) for x in X_test]
        all_tree_preds.append(preds)

    all_tree_preds = np.array(all_tree_preds)  # shape: (n_estimators, len(X_test))
    final_predictions = []

    # Deterministic majority voting
    for idx in range(X_test.shape[0]):
        test_sample_preds = all_tree_preds[:, idx]
        classes, counts = np.unique(test_sample_preds, return_counts=True)
        max_count = np.max(counts)
        best_classes = classes[counts == max_count]
        # Tie-breaker: choose smallest label
        final_predictions.append(int(np.min(best_classes)))

    return final_predictions
