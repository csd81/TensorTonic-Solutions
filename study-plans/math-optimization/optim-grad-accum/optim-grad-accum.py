import numpy as np

def gradient_accumulation(X: list, y: list, lr: float, micro_batch: int, accum_steps: int, n_epochs: int) -> dict:
    """
    Returns both loss curves and final weight vectors in a dictionary.
    """
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)
    N, D = X_arr.shape
    
    # Pre-split data into micro-batches for sequential, unshuffled processing
    micro_batches = []
    for i in range(0, N, micro_batch):
        X_b = X_arr[i:i + micro_batch]
        y_b = y_arr[i:i + micro_batch]
        micro_batches.append((X_b, y_b))
        
    def compute_full_mse(w):
        predictions = X_arr @ w
        return float(np.mean((predictions - y_arr) ** 2))

    # --- 1. Ordinary Method (No Accumulation) ---
    w_no_accum = np.zeros(D)
    no_accum_losses = []
    
    for epoch in range(n_epochs):
        for X_b, y_b in micro_batches:
            # Unnormalized gradient sum
            G_B = 2 * X_b.T @ (X_b @ w_no_accum - y_b)
            # Update immediately dividing by micro-batch sample count
            w_no_accum -= lr * (G_B / len(X_b))
        no_accum_losses.append(compute_full_mse(w_no_accum))
        
    # --- 2. Accumulated Method ---
    w_accum = np.zeros(D)
    accum_losses = []
    
    for epoch in range(n_epochs):
        accum_grad = np.zeros(D)
        accum_samples = 0
        step_count = 0
        
        for X_b, y_b in micro_batches:
            G_B = 2 * X_b.T @ (X_b @ w_accum - y_b)
            accum_grad += G_B
            accum_samples += len(X_b)
            step_count += 1
            
            # Update if window is full
            if step_count == accum_steps:
                w_accum -= lr * (accum_grad / accum_samples)
                # Reset window trackers
                accum_grad = np.zeros(D)
                accum_samples = 0
                step_count = 0
                
        # Flush remaining micro-batches at the end of the epoch if any
        if step_count > 0:
            w_accum -= lr * (accum_grad / accum_samples)
            
        accum_losses.append(compute_full_mse(w_accum))

    return {
        "no_accum_losses": no_accum_losses,
        "accum_losses": accum_losses,
        "no_accum_weights": w_no_accum.tolist(),
        "accum_weights": w_accum.tolist()
    }
