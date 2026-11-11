import numpy as np


def stratified_split(
    X: list, y: list, test_size: float = 0.2, seed: int = 42
) -> dict:
    """
    Returns a dictionary with X_train, X_test, y_train, and y_test.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    rng = np.random.default_rng(seed)
    classes = np.unique(y_arr)

    train_indices = []
    test_indices = []

    for cls in classes:
        cls_indices = np.where(y_arr == cls)[0]
        n_c = cls_indices.size

        # Shuffle this class's indices in place using the shared RNG
        shuffled = rng.permutation(cls_indices)

        # Compute test count for this class
        n_test = int(round(n_c * test_size))
        if n_c > 1:
            n_test = min(n_test, n_c - 1)
        else:
            n_test = 0

        test_indices.append(shuffled[:n_test])
        train_indices.append(shuffled[n_test:])

    # Combine and sort indices to preserve original dataset order
    train_idx = np.sort(np.concatenate(train_indices)).astype(np.int64)
    test_idx = np.sort(np.concatenate(test_indices)).astype(np.int64)

    return {
        "X_train": X_arr[train_idx],
        "X_test": X_arr[test_idx],
        "y_train": y_arr[train_idx],
        "y_test": y_arr[test_idx],
    }
