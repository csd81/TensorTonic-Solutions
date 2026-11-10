import numpy as np

def dbscan(X: list, eps: float = 0.5, min_samples: int = 5) -> list:
    """
    Implements DBSCAN clustering using Euclidean distance and a FIFO queue expansion.
    Returns integer cluster labels, using -1 for noise points.
    """
    # Convert input to a NumPy float64 array
    X = np.array(X, dtype=np.float64)
    n = len(X)
    
    # Initialize all labels as -2 (unvisited/unassigned)
    labels = [-2] * n
    
    # Precompute epsilon-neighborhoods for all points
    # N(p) includes the point itself and points on the boundary (<= eps)
    neighbors = []
    for i in range(n):
        dists = np.linalg.norm(X - X[i], axis=1)
        nb_indices = np.where(dists <= eps)[0]
        neighbors.append(sorted(list(nb_indices)))
        
    cluster_id = 0
    enqueued = [False] * n
    
    # Process points by ascending index
    for i in range(n):
        if labels[i] != -2:
            continue
            
        # If the point has fewer neighbors than min_samples, mark as noise
        if len(neighbors[i]) < min_samples:
            labels[i] = -1
            continue
            
        # Point is a core point; start a new cluster
        labels[i] = cluster_id
        enqueued[i] = True
        
        # Initialize FIFO queue with the core point's neighbors
        queue = []
        for nb_idx in neighbors[i]:
            if not enqueued[nb_idx]:
                queue.append(nb_idx)
                enqueued[nb_idx] = True
                
        # Expand the cluster using the FIFO queue
        head = 0
        while head < len(queue):
            q = queue[head]
            head += 1
            
            # Noise points reached by a core point join the cluster but do not expand it
            if labels[q] == -1:
                labels[q] = cluster_id
            elif labels[q] == -2:
                labels[q] = cluster_id
                
                # If the reached point is also a core point, expand its neighborhood
                if len(neighbors[q]) >= min_samples:
                    for nb_idx in neighbors[q]:
                        if not enqueued[nb_idx]:
                            queue.append(nb_idx)
                            enqueued[nb_idx] = True
                            
        cluster_id += 1
        
    return labels
