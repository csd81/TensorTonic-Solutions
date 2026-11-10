import numpy as np

def softmax_regression(X: list, y: list, num_classes: int, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight matrix and bias vector.
    """
    # Convert inputs to appropriate NumPy arrays
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    
    n_samples, n_features = X.shape
    
    # Initialize weights and biases to float64 zeros
    W = np.zeros((n_features, num_classes), dtype=np.float64)
    b = np.zeros(num_classes, dtype=np.float64)
    
    # Create one-hot encoded targets
    Y_one_hot = np.zeros((n_samples, num_classes), dtype=np.float64)
    Y_one_hot[np.arange(n_samples), y] = 1.0
    
    # Gradient Descent Optimization
    for _ in range(n_iters):
        # Forward pass: calculate logits
        Z = np.dot(X, W) + b
        
        # Stability adjustment: subtract row maximums
        Z_stable = Z - np.max(Z, axis=1, keepdims=True)
        
        # Compute row-wise stable softmax probabilities
        exp_Z = np.exp(Z_stable)
        P = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
        # Compute the gradient of cross-entropy loss w.r.t logits (divided by N for batch average)
        dZ = (P - Y_one_hot) / n_samples
        
        # Gradients w.r.t parameters
        dW = np.dot(X.T, dZ)
        db = np.sum(dZ, axis=0)
        
        # Parameter updates
        W -= lr * dW
        b -= lr * db
        
    # Round to 4 decimal places and remove negative zeros (-0.0 -> 0.0)
    W_rounded = (np.round(W, 4) + 0.0).tolist()
    b_rounded = (np.round(b, 4) + 0.0).tolist()
    
    return W_rounded, b_rounded
