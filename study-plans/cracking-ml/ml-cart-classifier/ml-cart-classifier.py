from fractions import Fraction
import numpy as np

def cart_classify(X_train: list, y_train: list, X_test: list, max_depth: int = 5, min_samples: int = 2) -> list:
    """
    Returns one predicted class label per test row using a CART decision tree.
    """
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.int64)
    X_test = np.array(X_test, dtype=np.float64)
    
    # Exact Gini Impurity calculation using Fraction
    def get_gini(labels):
        n = len(labels)
        if n == 0:
            return Fraction(0, 1)
        _, counts = np.unique(labels, return_counts=True)
        return Fraction(1, 1) - sum(Fraction(int(count) ** 2, n ** 2) for count in counts)

    def get_majority_class(labels):
        if len(labels) == 0:
            return 0
        classes, counts = np.unique(labels, return_counts=True)
        return classes[np.argmax(counts)]

    class DecisionNode:
        def __init__(self, depth):
            self.depth = depth
            self.is_leaf = False
            self.feature = None
            self.threshold = None
            self.left = None
            self.right = None
            self.prediction = None

    def build_tree(X, y, depth):
        node = DecisionNode(depth)
        
        # Stopping criteria
        if depth >= max_depth or len(y) < min_samples or len(np.unique(y)) == 1:
            node.is_leaf = True
            node.prediction = get_majority_class(y)
            return node
            
        # Using Fraction(0, 1) as baseline for positive reduction
        best_gain = Fraction(0, 1)
        best_feat = None
        best_thresh = None
        
        n_samples, n_features = X.shape
        current_gini = get_gini(y)
        
        for j in range(n_features):
            unique_thresholds = np.unique(X[:, j])
            for t in unique_thresholds:
                left_mask = X[:, j] <= t
                right_mask = ~left_mask
                
                # A split must actually separate data to be valid
                if not np.any(left_mask) or np.all(left_mask):
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                w_left = Fraction(len(y_left), n_samples)
                w_right = Fraction(len(y_right), n_samples)
                
                # Compute exact split impurity
                gain = current_gini - (w_left * get_gini(y_left) + w_right * get_gini(y_right))
                
                # strict inequality captures the first split with the greatest positive reduction
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = t
                    
        if best_feat is None:
            node.is_leaf = True
            node.prediction = get_majority_class(y)
            return node
            
        node.feature = best_feat
        node.threshold = best_thresh
        
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        
        node.left = build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    root = build_tree(X_train, y_train, 0)
    
    def traverse(node, x):
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.prediction

    return [int(traverse(root, sample)) for sample in X_test]
