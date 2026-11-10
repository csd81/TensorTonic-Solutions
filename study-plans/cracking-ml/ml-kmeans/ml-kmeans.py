import numpy as np

def kmeans(X: list, k: int, max_iters: int = 100, seed: int = 42) -> tuple:
    # Convert input to a float64 NumPy array
    X_arr = np.array(X, dtype=np.float64)
    n_samples, n_features = X_arr.shape
    
    # Initialize the random number generator and pick k distinct row indices
    rng = np.random.RandomState(seed)
    initial_indices = rng.choice(n_samples, size=k, replace=False)
    centroids = X_arr[initial_indices].copy()
    
    for _ in range(max_iters):
        # Compute squared Euclidean distances from each point to each centroid
        # shape: (n_samples, k)
        distances = np.sum((X_arr[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        
        # Assign points to closest centroid. np.argmin naturally breaks ties 
        # by choosing the lowest centroid index.
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.copy(centroids)
        for j in range(k):
            # Find all points belonging to cluster j
            cluster_points = X_arr[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = np.mean(cluster_points, axis=0)
            # If empty, the centroid remains unchanged (keeps its preceding value)
        
        # Check convergence condition
        if np.allclose(centroids, new_centroids):
            centroids = new_centroids
            break
            
        centroids = new_centroids
        
    # Recompute final assignments one last time from the final centroids
    final_distances = np.sum((X_arr[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
    final_labels = np.argmin(final_distances, axis=1)
    
    # Round outputs to 4 decimal places and convert to nested lists
    rounded_centroids = np.round(centroids, 4).tolist()
    final_labels_list = final_labels.astype(int).tolist()
    
    return final_labels_list, rounded_centroids
