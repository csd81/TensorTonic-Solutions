import numpy as np

def kmeans_plusplus(X: list, k: int, seed: int = 42) -> list:
    """
    Returns the selected centroid rows in selection order.
    """
    # 1. Convert input to a NumPy float64 array
    X_arr = np.array(X, dtype=np.float64)
    n, d = X_arr.shape
    
    # Initialize the RandomState
    rng = np.random.RandomState(seed)
    
    # 2. Choose the first row index using rng.randint
    first_idx = rng.randint(0, n)
    selected_indices = [first_idx]
    
    # Array to track the minimum squared distance from each point to its nearest selected centroid
    # Start with infinity so any distance calculated is smaller
    min_dist_sq = np.full(n, np.inf)
    
    # 3. Iteratively select the remaining k-1 centroids
    while len(selected_indices) < k:
        latest_centroid = X_arr[selected_indices[-1]]
        
        # Calculate squared distance from all points to the newest centroid
        # Using axis=1 to sum over features/dimensions
        dist_sq = np.sum((X_arr - latest_centroid) ** 2, axis=1)
        
        # Update the nearest-centroid squared distances
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)
        
        # 4. Sample each later row using rng.choice and the normalized distance probabilities
        total_dist_sq = np.sum(min_dist_sq)
        probabilities = min_dist_sq / total_dist_sq
        
        next_idx = rng.choice(n, p=probabilities)
        selected_indices.append(next_idx)
        
    # 5. Extract rows, round numeric outputs to 4 decimal places, and convert to nested lists
    centroids = X_arr[selected_indices]
    rounded_centroids = np.round(centroids, 4).tolist()
    
    return rounded_centroids
