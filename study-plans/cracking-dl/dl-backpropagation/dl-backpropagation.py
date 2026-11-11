import numpy as np

def backprop(X: list, y: list, W1: list, b1: list, W2: list, b2: list) -> dict:
    """
    Returns gradients for both weight matrices and bias vectors.
    """
    # 1. Convert all inputs to NumPy float64 arrays
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    b1 = np.array(b1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    b2 = np.array(b2, dtype=np.float64)
    
    n = X.shape[0]
    
    # 2. Forward Pass
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)  # ReLU activation
    Y_hat = A1 @ W2 + b2
    
    # 3. Backward Pass
    # MSE loss gradient w.r.t Y_hat (includes the 2/n factor)
    dY_hat = (2.0 / n) * (Y_hat - y)
    
    # Gradients for Output Layer
    dW2 = A1.T @ dY_hat
    db2 = np.sum(dY_hat, axis=0)
    
    # Gradients for Hidden Layer
    dA1 = dY_hat @ W2.T
    dZ1 = dA1 * (Z1 > 0)    # ReLU derivative (0 where Z1 <= 0)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)
    
    # 4. Format and round nested lists to 4 decimal places
    return {
        "dW1": np.round(dW1, 4).tolist(),
        "db1": np.round(db1, 4).tolist(),
        "dW2": np.round(dW2, 4).tolist(),
        "db2": np.round(db2, 4).tolist()
    }
