import numpy as np


def kfold_split(N: int, k: int, shuffle: bool = True, seed: int = 0) -> list:
    """
    Returns a list of dictionaries with train_idx and val_idx.
    """
    indices = np.arange(N, dtype=np.int64)

    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)

    # np.array_split automatically assigns 1 extra element to the first (N % k) folds
    folds = np.array_split(indices, k)

    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_folds = [folds[j] for j in range(k) if j != i]
        train_idx = np.concatenate(train_folds) if train_folds else np.array([], dtype=np.int64)

        splits.append(
            {
                "train_idx": train_idx.astype(np.int64),
                "val_idx": val_idx.astype(np.int64),
            }
        )

    return splits
