import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    train = np.asarray(X_train, dtype=np.float64)
    test = np.asarray(X_test, dtype=np.float64)

    # Reshape 1D scalar sample inputs into 2D (n_samples, 1)
    if train.ndim == 1:
        train = train.reshape(-1, 1)
    if test.ndim == 1:
        test = test.reshape(-1, 1)

    n_train = train.shape[0]
    n_test = test.shape[0]

    # Compute pairwise Euclidean distances of shape (n_test, n_train)
    diff = test[:, np.newaxis, :] - train[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Use mergesort (stable sort) so tied distances keep ascending index order
    sorted_indices = np.argsort(dists, axis=1, kind="mergesort")

    # Select up to k nearest neighbors
    limit = min(k, n_train)
    nearest = sorted_indices[:, :limit]

    # Pad with -1 if k exceeds the training set size
    if k > n_train:
        pad = np.full((n_test, k - n_train), -1, dtype=np.int64)
        nearest = np.hstack((nearest, pad))

    return nearest.astype(np.int64)
