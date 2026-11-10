import numpy as np

def isolation_forest(X: list, n_estimators: int = 100, max_samples: int = 256, seed: int = 42) -> list:
    """
    Returns one anomaly score per input row using the Isolation Forest algorithm.
    """
    X_arr = np.array(X, dtype=float)
    n = len(X_arr)
    sample_size = min(max_samples, n)
    max_depth = int(np.ceil(np.log2(sample_size)))
    rng = np.random.RandomState(seed)
    
    def get_c(s: int) -> float:
        if s <= 1: 
            return 0.0
        if s == 2: 
            return 1.0
        return 2.0 * (np.log(s - 1) + 0.5772156649) - 2.0 * (s - 1) / s

    def build_tree(X_sub: np.ndarray, depth: int) -> dict:
        if depth >= max_depth or len(X_sub) <= 1:
            return {'leaf': True, 'size': len(X_sub), 'depth': depth}
            
        mins = X_sub.min(axis=0)
        maxs = X_sub.max(axis=0)
        valid_features = np.where(maxs > mins)[0]
        
        if len(valid_features) == 0:
            return {'leaf': True, 'size': len(X_sub), 'depth': depth}
            
        feat = rng.choice(valid_features)
        threshold = rng.uniform(mins[feat], maxs[feat])
        left_mask = X_sub[:, feat] < threshold
        
        return {
            'leaf': False, 
            'feat': feat, 
            'threshold': threshold,
            'left': build_tree(X_sub[left_mask], depth + 1),
            'right': build_tree(X_sub[~left_mask], depth + 1)
        }

    def get_path_length(x: np.ndarray, node: dict) -> float:
        if node['leaf']:
            return node['depth'] + get_c(node['size'])
        if x[node['feat']] < node['threshold']:
            return get_path_length(x, node['left'])
        return get_path_length(x, node['right'])

    # Accumulate path lengths for all input observations across all trees
    total_paths = np.zeros(n)
    for _ in range(n_estimators):
        subsample_indices = rng.choice(n, sample_size, replace=False)
        tree = build_tree(X_arr[subsample_indices], 0)
        for i in range(n):
            total_paths[i] += get_path_length(X_arr[i], tree)
            
    avg_paths = total_paths / n_estimators
    c_factor = get_c(sample_size)
    
    # Avoid division by zero if c_factor is 0
    if c_factor == 0:
        scores = np.zeros(n)
    else:
        scores = 2 ** (-avg_paths / c_factor)
        
    return [round(float(s), 4) for s in scores]
