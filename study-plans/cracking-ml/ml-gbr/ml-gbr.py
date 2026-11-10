import numpy as np

class CARTRegressorNode:
    def __init__(self, depth: int, max_depth: int):
        self.depth = depth
        self.max_depth = max_depth
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

    def fit(self, X: np.ndarray, r: np.ndarray):
        # Base case: max depth reached, insufficient samples, or constant residuals
        if self.depth >= self.max_depth or len(X) <= 1 or np.all(r == r[0]):
            self.value = np.mean(r)
            return
        
        best_reduction = 0.0
        best_feat = None
        best_thresh = None
        
        n_samples, n_features = X.shape
        initial_loss = np.sum((r - np.mean(r)) ** 2)
        
        # Search features and observed thresholds in ascending order
        for j in range(n_features):
            thresholds = sorted(list(set(X[:, j])))
            for t in thresholds:
                left_mask = X[:, j] <= t
                right_mask = ~left_mask
                
                if not np.any(left_mask) or not np.any(right_mask):
                    continue
                
                r_left = r[left_mask]
                r_right = r[right_mask]
                
                loss_left = np.sum((r_left - np.mean(r_left)) ** 2)
                loss_right = np.sum((r_right - np.mean(r_right)) ** 2)
                
                reduction = initial_loss - (loss_left + loss_right)
                
                # Select the first strictly best positive MSE reduction
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feat = j
                    best_thresh = t
                    
        # If a valid split is found, recursively build children
        if best_reduction > 0:
            self.feature = best_feat
            self.threshold = best_thresh
            
            left_mask = X[:, best_feat] <= best_thresh
            self.left = CARTRegressorNode(self.depth + 1, self.max_depth)
            self.left.fit(X[left_mask], r[left_mask])
            
            self.right = CARTRegressorNode(self.depth + 1, self.max_depth)
            self.right.fit(X[~left_mask], r[~left_mask])
        else:
            self.value = np.mean(r)

    def predict_row(self, x: np.ndarray) -> float:
        if self.value is not None:
            return self.value
        if x[self.feature] <= self.threshold:
            return self.left.predict_row(x)
        else:
            return self.right.predict_row(x)


def gbr_predict(
    X_train: list, 
    y_train: list, 
    X_test: list, 
    n_estimators: int = 10, 
    max_depth: int = 3, 
    learning_rate: float = 0.1, 
    seed: int = 42
) -> list:
    """
    Returns one boosted regression prediction per test row.
    """
    # Convert inputs to float64 numpy arrays
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    
    # Initialize ensemble predictions with the mean of target targets
    initial_mean = np.mean(y_train)
    F_train = np.full(y_train.shape, initial_mean)
    F_test = np.full(len(X_test), initial_mean)
    
    # Sequentially fit each tree on the negative gradient (residuals)
    for _ in range(n_estimators):
        r = y_train - F_train
        
        tree = CARTRegressorNode(depth=0, max_depth=max_depth)
        tree.fit(X_train, r)
        
        # Compute current tree predictions for both train and test sets
        h_train = np.array([tree.predict_row(x) for x in X_train])
        h_test = np.array([tree.predict_row(x) for x in X_test])
        
        # Update ensemble predictions
        F_train += learning_rate * h_train
        F_test += learning_rate * h_test
        
    return [round(x, 4) for x in F_test]
