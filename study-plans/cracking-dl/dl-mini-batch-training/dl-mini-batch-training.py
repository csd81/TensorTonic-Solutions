import numpy as np

def mini_batch_training(X: list, y: list, weights: list, biases: list, lr: float, epochs: int, batch_size: int) -> list:
    """
    Returns the full-dataset loss after each epoch.
    """
    # 1. Convert to float64 arrays and copy weights/biases to protect original inputs
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    W_list = [np.asarray(w, dtype=np.float64).copy() for w in weights]
    b_list = [np.asarray(b, dtype=np.float64).copy() for b in biases]
    
    num_layers = len(W_list)
    N = X_arr.shape[0]
    epoch_losses = []
    
    def forward(X_batch):
        # activations keeps batches as rows (batch_size, features)
        activations = [X_batch]
        pre_activations = []
        
        for l in range(num_layers):
            # z = X_prev @ W^T + b
            z = activations[-1] @ W_list[l].T + b_list[l]
            pre_activations.append(z)
            
            # Hidden layer activations get ReLU; the output layer remains linear
            if l < num_layers - 1:
                activations.append(np.maximum(0.0, z))
            else:
                activations.append(z)
        return activations, pre_activations

    # 2. Main Training Loop
    for epoch in range(epochs):
        for start in range(0, N, batch_size):
            X_batch = X_arr[start : start + batch_size]
            y_batch = y_arr[start : start + batch_size]
            m = len(X_batch)
            
            # Forward pass
            activations, pre_activations = forward(X_batch)
            
            # Backpropagation using row-major orientation
            delta = (activations[-1] - y_batch) / m
            
            grad_w = [None] * num_layers
            grad_b = [None] * num_layers
            
            for l in reversed(range(num_layers)):
                grad_w[l] = delta.T @ activations[l]
                grad_b[l] = np.sum(delta, axis=0)
                
                if l > 0:
                    delta = (delta @ W_list[l]) * (pre_activations[l-1] > 0.0)
            
            # Parameter updates
            for l in range(num_layers):
                W_list[l] -= lr * grad_w[l]
                b_list[l] -= lr * grad_b[l]
        
        # 3. Evaluate full-dataset loss at the end of the epoch
        predictions = forward(X_arr)[0][-1]
        loss = 0.5 * np.mean(np.sum((predictions - y_arr) ** 2, axis=1))
        epoch_losses.append(round(float(loss), 4))
        
    return epoch_losses
