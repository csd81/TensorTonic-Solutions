import numpy as np

def batch_generator(X: list, y: list, batch_size: int, seed: int = 42, drop_last: bool = False):
    """
    Returns a generator of (X_batch, y_batch) tuples.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    n_samples = len(X_arr)
    
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    
    X_shuffled = X_arr[indices]
    y_shuffled = y_arr[indices]
    
    limit = (n_samples // batch_size) * batch_size if drop_last else n_samples
    
    for start in range(0, limit, batch_size):
        yield (X_shuffled[start:start + batch_size], y_shuffled[start:start + batch_size])
