import numpy as np

def stacking_classify(X_train: list, y_train: list, X_test: list, n_folds: int = 3, seed: int = 42) -> list:
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.int64)
    X_test = np.array(X_test, dtype=np.float64)
    
    classes = np.sort(np.unique(y_train))
    
    # 1. Exact fold creation with NumPy RandomState shuffle
    rng = np.random.RandomState(seed)
    indices = np.arange(len(X_train))
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    # Gini Impurity Helper
    def get_gini(labels):
        if len(labels) == 0:
            return 0.0
        counts = np.bincount(labels)
        probs = counts / len(labels)
        return 1.0 - np.sum(probs**2)
        
    # Fit Gini Decision Stump
    def fit_stump(X, y):
        N, D = X.shape
        best_gain = 0.0
        best_col = -1
        best_thresh = None
        parent_gini = get_gini(y)
        
        for j in range(D):
            vals = X[:, j]
            unique_vals = np.sort(np.unique(vals))
            for t in unique_vals:
                left_mask = vals <= t
                right_mask = ~left_mask
                if not np.any(left_mask) or not np.any(right_mask):
                    continue
                
                y_l, y_r = y[left_mask], y[right_mask]
                gain = parent_gini - (len(y_l)/N) * get_gini(y_l) - (len(y_r)/N) * get_gini(y_r)
                if gain > best_gain:
                    best_gain = gain
                    best_col = j
                    best_thresh = t
                    
        if best_gain > 0:
            vals = X[:, best_col]
            left_mask = vals <= best_thresh
            pred_left = np.argmax(np.bincount(y[left_mask]))
            pred_right = np.argmax(np.bincount(y[~left_mask]))
            return (best_col, best_thresh, pred_left, pred_right)
        else:
            return (None, None, np.argmax(np.bincount(y)), None)

    def predict_stump(stump, X):
        col, thresh, pred_left, pred_right = stump
        if col is None:
            return np.full(len(X), pred_left, dtype=np.int64)
        return np.where(X[:, col] <= thresh, pred_left, pred_right)

    # 3-NN Classifier Prediction
    def predict_knn(X_tr, y_tr, X_te):
        preds = []
        for x in X_te:
            dists = np.sum((X_tr - x)**2, axis=1)
            idx = np.argsort(dists, kind='stable')[:3]
            preds.append(np.argmax(np.bincount(y_tr[idx])))
        return np.array(preds, dtype=np.int64)

    # 2. Out-of-fold feature generation
    OOF = np.zeros((len(X_train), 2), dtype=np.float64)
    for i in range(n_folds):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val = X_train[val_idx]
        
        stump = fit_stump(X_tr, y_tr)
        OOF[val_idx, 0] = predict_stump(stump, X_val)
        OOF[val_idx, 1] = predict_knn(X_tr, y_tr, X_val)
        
    # 3. Refit base models on all original training data
    stump_full = fit_stump(X_train, y_train)
    X_test_meta = np.column_stack([
        predict_stump(stump_full, X_test),
        predict_knn(X_train, y_train, X_test)
    ])
    
    # 4. Train One-Vs-Rest Meta-Model (Logistic Regression)
    meta_scores = []
    N_samples = len(y_train)
    for c in classes:
        y_binary = (y_train == c).astype(np.float64)
        w = np.zeros(2)
        b = 0.0
        
        for _ in range(500):
            z = OOF @ w + b
            z_clipped = np.clip(z, -500.0, 500.0)
            pred_prob = 1.0 / (1.0 + np.exp(-z_clipped))
            
            grad_w = (OOF.T @ (pred_prob - y_binary)) / N_samples
            grad_b = np.sum(pred_prob - y_binary) / N_samples
            
            w -= 0.1 * grad_w
            b -= 0.1 * grad_b
            
        # Logits score calculation for test set
        test_z = X_test_meta @ w + b
        meta_scores.append(test_z)
        
    # 5. Determine final predictions breaking ties with the first class label
    meta_scores = np.column_stack(meta_scores)
    final_pred_indices = np.argmax(meta_scores, axis=1)
    
    return [int(classes[idx]) for idx in final_pred_indices]
