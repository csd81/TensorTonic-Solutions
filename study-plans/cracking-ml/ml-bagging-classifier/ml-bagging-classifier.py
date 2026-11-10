import numpy as np

def gini_impurity(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)

class CARTTree:
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.tree = None

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

        n_samples, n_features = X.shape
        parent_gini = gini_impurity(y)
        best_gain = -1.0
        best_split = None

        for feature_idx in range(n_features):
            thresholds = np.sort(np.unique(X[:, feature_idx]))
            for thr in thresholds:
                left_mask = X[:, feature_idx] <= thr
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                gain = parent_gini - (len(y_left) / n_samples) * gini_impurity(y_left) - (len(y_right) / n_samples) * gini_impurity(y_right)

                # Update only if the Gini gain is strictly larger
                if gain > best_gain + 1e-11:
                    best_gain = gain
                    best_split = {
                        'feature': feature_idx,
                        'threshold': thr,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }

        if best_split is None or best_gain <= 0.0:
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

def bagging_classify(X_train: list, y_train: list, X_test: list, n_estimators: int = 10, max_depth: int = 5, seed: int = 42) -> list:
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    n = len(X_train)
    
    rng = np.random.RandomState(seed)
    all_tree_preds = []

    for _ in range(n_estimators):
        # Draw bootstrap sample indices
        bootstrap_indices = rng.randint(0, n, size=n)
        X_b = X_train[bootstrap_indices]
        y_b = y_train[bootstrap_indices]

        # Train a base CART tree
        tree = CARTTree(max_depth=max_depth)
        root_node = tree.fit(X_b, y_b)

        # Generate predictions for the entire test set
        preds = [tree.predict_row(root_node, x) for x in X_test]
        all_tree_preds.append(preds)

    all_tree_preds = np.array(all_tree_preds)  # shape: (n_estimators, len(X_test))
    final_predictions = []

    # Deterministic majority vote per test sample
    for idx in range(X_test.shape[0]):
        test_sample_preds = all_tree_preds[:, idx]
        classes, counts = np.unique(test_sample_preds, return_counts=True)
        max_count = np.max(counts)
        best_classes = classes[counts == max_count]
        # Choose the smallest label on a tie
        final_predictions.append(int(np.min(best_classes)))

    return final_predictions
