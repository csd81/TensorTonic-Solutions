import numpy as np

def batch_gd_compare(X: list, y: list, batch_sizes: list, n_epochs: int, lr: float, seed: int) -> list:
    """
    Returns one post-epoch MSE loss curve per batch size.
    """
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)
    
    n_samples, n_features = X_arr.shape
    results = []
    
    for b_size in batch_sizes:
        # Reset weights and random generator for each batch size run
        w = np.zeros(n_features, dtype=float)
        rng = np.random.RandomState(seed)
        
        loss_curve = []
        
        for epoch in range(n_epochs):
            # Shuffle rows once at the beginning of each epoch
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            
            # Process consecutive mini-batches
            for i in range(0, n_samples, b_size):
                batch_idx = indices[i:i + b_size]
                
                X_batch = X_arr[batch_idx]
                y_batch = y_arr[batch_idx]
                B = len(batch_idx)
                
                # Compute gradient for the current batch
                # g = (2 / B) * X_B^T * (X_B * w - y_B)
                predictions = X_batch @ w
                error = predictions - y_batch
                gradient = (2.0 / B) * (X_batch.T @ error)
                
                # Update weights
                w -= lr * gradient
                
            # Compute and record the full-dataset MSE after all updates in this epoch
            full_preds = X_arr @ w
            full_mse = float(np.mean((full_preds - y_arr) ** 2))
            loss_curve.append(full_mse)
            
        results.append(loss_curve)
        
    return results
