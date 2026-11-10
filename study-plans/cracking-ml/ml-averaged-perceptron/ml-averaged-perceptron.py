import numpy as np

def averaged_perceptron(X: list, y: list, lr: float, epochs: int) -> dict:
    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    
    n_samples, n_features = X_arr.shape
    
    # Initialize active weights and bias
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0
    
    # Accumulators for the averaged parameters
    w_sum = np.zeros(n_features, dtype=np.float64)
    b_sum = 0.0
    
    # Total number of iterations across all epochs
    total_steps = 0
    
    # Sequentially pass through the samples for exactly `epochs` times
    for epoch in range(epochs):
        for i in range(n_samples):
            x_i = X_arr[i]
            y_i = y_arr[i]
            
            # Check the activation margin condition: y_i * (x_i^T w + b) <= 0
            if y_i * (np.dot(x_i, w) + b) <= 0:
                w += lr * y_i * x_i
                b += lr * y_i
            
            # The average must include the state after every sample visit
            w_sum += w
            b_sum += b
            total_steps += 1
            
    # Calculate final averaged parameters
    w_avg = w_sum / total_steps
    b_avg = b_sum / total_steps
    
    # Compute predictions using the averaged parameters
    # Predict +1 for strictly positive score, -1 otherwise
    scores = np.dot(X_arr, w_avg) + b_avg
    predictions = np.where(scores > 0, 1, -1).tolist()
    
    return {
        "bias": round(float(b_avg), 4),
        "predictions": [int(p) for p in predictions],
        "weights": [round(float(val), 4) for val in w_avg.tolist()]
    }
