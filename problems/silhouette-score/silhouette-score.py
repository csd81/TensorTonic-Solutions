import numpy as np

def silhouette_score(X: list, labels: list[int]) -> float:
    """
    Returns the mean Silhouette Score as a Python float.
    """
    X_arr = np.asarray(X, dtype=float)
    labels_arr = np.asarray(labels)
    n_samples = X_arr.shape[0]

    # Pairwise Euclidean distance matrix of shape (N, N)
    diff = X_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    unique_labels = np.unique(labels_arr)
    a = np.zeros(n_samples, dtype=float)
    b = np.full(n_samples, np.inf, dtype=float)

    for label in unique_labels:
        in_cluster = labels_arr == label
        cluster_size = np.sum(in_cluster)

        # Intra-cluster mean distance excluding self (diagonal distance is 0)
        intra_dists = dist_matrix[np.ix_(in_cluster, in_cluster)]
        a[in_cluster] = np.sum(intra_dists, axis=1) / (cluster_size - 1)

        # Nearest other-cluster mean distance
        for other_label in unique_labels:
            if other_label == label:
                continue
            in_other = labels_arr == other_label
            inter_dists = dist_matrix[np.ix_(in_cluster, in_other)]
            mean_inter = np.mean(inter_dists, axis=1)
            b[in_cluster] = np.minimum(b[in_cluster], mean_inter)

    denom = np.maximum(a, b)
    s = np.where(denom > 0.0, (b - a) / denom, 0.0)

    return float(np.mean(s))
