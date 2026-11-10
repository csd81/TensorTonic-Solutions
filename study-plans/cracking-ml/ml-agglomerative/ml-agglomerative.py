import numpy as np

def agglomerative(X: list, n_clusters: int = 2, linkage: str = "single") -> list:
    """
    Returns one hierarchical-cluster label per row.
    """
    X_arr = np.array(X, dtype=np.float64)
    n_samples = len(X_arr)
    
    # Initialize clusters: each point starts in its own cluster with its index as the ID
    # Map from cluster_id -> list of original sample indices belonging to it
    clusters = {i: [i] for i in range(n_samples)}
    
    # Precompute a pairwise Euclidean distance matrix for points
    # (Since N is typically small in these types of challenges, this is efficient)
    dist_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = np.linalg.norm(X_arr[i] - X_arr[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    def compute_linkage_distance(c1_id, c2_id):
        indices1 = clusters[c1_id]
        indices2 = clusters[c2_id]
        
        if linkage == "single":
            return min(dist_matrix[i, j] for i in indices1 for j in indices2)
        elif linkage == "complete":
            return max(dist_matrix[i, j] for i in indices1 for j in indices2)
        elif linkage == "average":
            total_dist = sum(dist_matrix[i, j] for i in indices1 for j in indices2)
            return total_dist / (len(indices1) * len(indices2))
        return 0.0

    # Merge until we hit the target number of clusters
    while len(clusters) > n_clusters:
        best_tuple = None
        cluster_ids = sorted(list(clusters.keys()))
        
        # Find the minimum candidate tuple: (distance, smaller_id, larger_id)
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                c1 = cluster_ids[i]
                c2 = cluster_ids[j]
                
                d = compute_linkage_distance(c1, c2)
                candidate = (d, c1, c2)
                
                if best_tuple is None or candidate < best_tuple:
                    best_tuple = candidate
                    
        # Perform the merge based on the best candidate tuple found
        _, min_id, max_id = best_tuple
        clusters[min_id].extend(clusters[max_id])
        del clusters[max_id]

    # Map surviving cluster IDs to sequential labels starting from 0 in ascending order
    surviving_ids = sorted(list(clusters.keys()))
    id_to_label = {cluster_id: label for label, cluster_id in enumerate(surviving_ids)}
    
    # Assign the correct label back to each original row index
    output_labels = [0] * n_samples
    for cluster_id, original_indices in clusters.items():
        label = id_to_label[cluster_id]
        for idx in original_indices:
            output_labels[idx] = label
            
    return output_labels
