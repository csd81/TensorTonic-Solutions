from fractions import Fraction
import numpy as np

def cart_regress(X_train: list, y_train: list, X_test: list, max_depth: int = 5, min_samples: int = 2) -> list:
    """
    Returns one regression prediction per test row, rounded to 4 decimal places.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    
    # Exact MSE Impurity calculation using Fraction
    def get_mse_exact(values):
        if len(values) == 0:
            return Fraction(0, 1)
        exact = [Fraction(float(value)) for value in values]
        mean = sum(exact) / len(exact)
        return sum((value - mean) ** 2 for value in exact) / len(exact)

    class RegressionNode:
        def __init__(self, depth):
            self.depth = depth
            self.is_leaf = False
            self.feature = None
            self.threshold = None
            self.left = None
            self.right = None
            self.prediction = None

    def build_tree(X, values, depth):
        node = RegressionNode(depth)
        
        # Stopping criteria
        if depth >= max_depth or len(values) < min_samples or len(np.unique(values)) == 1:
            node.is_leaf = True
            node.prediction = float(np.mean(values)) if len(values) > 0 else 0.0
            return node
            
        # Using pure 0 Fraction as strict positive reduction threshold
        best_gain = Fraction(0, 1)
        best_feat = None
        best_thresh = None
        
        n_samples, n_features = X.shape
        parent_mse = get_mse_exact(values)
        
        for j in range(n_features):
            unique_thresholds = np.unique(X[:, j])
            for t in unique_thresholds:
                left_mask = X[:, j] <= t
                
                # Split must separate data to be valid
                if not np.any(left_mask) or np.all(left_mask):
                    continue
                    
                values_left = values[left_mask]
                values_right = values[~left_mask]
                
                w_left = Fraction(len(values_left), n_samples)
                w_right = Fraction(len(values_right), n_samples)
                
                gain = parent_mse - (w_left * get_mse_exact(values_left) + w_right * get_mse_exact(values_right))
                
                # Use strict inequality to capture the first split maximizing positive reduction
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = t
                    
        if best_feat is None:
            node.is_leaf = True
            node.prediction = float(np.mean(values))
            return node
            
        node.feature = best_feat
        node.threshold = best_thresh
        
        left_mask = X[:, best_feat] <= best_thresh
        
        node.left = build_tree(X[left_mask], values[left_mask], depth + 1)
        node.right = build_tree(X[~left_mask], values[~left_mask], depth + 1)
        return node

    root = build_tree(X_train, y_train, 0)
    
    def traverse(node, x):
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.prediction

    return [round(float(traverse(root, sample)), 4) for sample in X_test]
