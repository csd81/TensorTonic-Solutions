import numpy as np

def e_means(X: list, centroids: list, batches: list, alpha: float) -> dict:
    """
    Returns a dict with the final centroids (k, d), the running counts (k,), 
    the labels (n,) and the mse.
    """
    X = np.array(X, dtype=float)
    c = np.array(centroids, dtype=float)
    k, d = c.shape
    n = X.shape[0]
    
    # Initialize running sum (S) and counts (N)
    S = c.copy()
    N = np.ones(k, dtype=float)
    
    for batch in batches:
        batch_X = X[batch]
        
        # Step 1: Assign each point in the mini-batch to its nearest centroid
        # Broadcast to compute squared Euclidean distance of shape (len(batch), k)
        dists = np.sum((batch_X[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=-1)
        
        # np.argmin automatically breaks ties toward the smaller index
        batch_labels = np.argmin(dists, axis=1)
        
        # Step 2: Compute mini-batch sums (s) and counts (n) per cluster
        s = np.zeros_like(S)
        counts_b = np.zeros_like(N)
        for j in range(k):
            mask = (batch_labels == j)
            counts_b[j] = np.sum(mask)
            if counts_b[j] > 0:
                s[j] = np.sum(batch_X[mask], axis=0)
                
        # Step 3: Update running statistics using Exponential Moving Average
        S = (1 - alpha) * S + alpha * s
        N = (1 - alpha) * N + alpha * counts_b
        
        # Step 4: Re-compute centroids before the next batch is processed
        for j in range(k):
            if N[j] > 0:
                c[j] = S[j] / N[j]
                
    # Step 5: Perform final assignment of all points using final centroids
    dists_final = np.sum((X[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=-1)
    labels = np.argmin(dists_final, axis=1)
    
    # Step 6: Compute the final Mean Squared Error (MSE)
    min_dists = dists_final[np.arange(n), labels]
    mse = float(np.mean(min_dists))
    
    return {
        "centroids": c,
        "counts": N,
        "labels": labels,
        "mse": mse
    }
