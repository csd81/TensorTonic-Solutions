import numpy as np

def dropout_mlp(X_train: list, y_train: list, X_test: list, y_test: list, hidden_size: int, lr: float, dropout_rate: float, n_epochs: int, seed: int) -> list:
    """
    Returns one post-update test accuracy per epoch.
    """
    # Set seed once before initialization
    np.random.seed(seed)
    
    # Convert input lists to NumPy arrays
    X_tr = np.array(X_train, dtype=float)
    y_tr = np.array(y_train, dtype=float).reshape(-1, 1)
    X_te = np.array(X_test, dtype=float)
    y_te = np.array(y_test, dtype=float).reshape(-1, 1)
    
    m_tr = X_tr.shape[0]
    m_te = X_te.shape[0]
    input_dim = X_tr.shape[1]
    
    # 1. Initialize weights with He/Kaiming initialization scaling
    W1 = np.random.randn(input_dim, hidden_size) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, 1))
    
    test_accuracies = []
    
    for epoch in range(n_epochs):
        # --- Forward Propagation (Train) ---
        Z1 = X_tr @ W1 + b1
        A1 = np.maximum(0, Z1)  # ReLU activation
        
        # Apply Inverted Dropout Mask
        if dropout_rate > 0.0:
            mask = (np.random.rand(*A1.shape) > dropout_rate).astype(float)
            A1 = (A1 * mask) / (1.0 - dropout_rate)
        else:
            mask = np.ones_like(A1)
            
        Z2 = A1 @ W2 + b2
        A2 = 1.0 / (1.0 + np.exp(-np.clip(Z2, -500, 500)))  # Sigmoid activation
        
        # --- Backpropagation ---
        dZ2 = (A2 - y_tr) / m_tr
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = dZ2 @ W2.T
        # Reuse the exact dropout mask and inverted scaling factor in backprop
        if dropout_rate > 0.0:
            dA1 = (dA1 * mask) / (1.0 - dropout_rate)
            
        dZ1 = dA1 * (Z1 > 0)  # ReLU gradient
        dW1 = X_tr.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # --- Gradient Descent Update ---
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        
        # --- Evaluation on Test Set (No Dropout) ---
        Z1_te = X_te @ W1 + b1
        A1_te = np.maximum(0, Z1_te)
        Z2_te = A1_te @ W2 + b2
        A2_te = 1.0 / (1.0 + np.exp(-np.clip(Z2_te, -500, 500)))
        
        # Map probabilities to absolute classes (threshold at 0.5)
        predictions = (A2_te >= 0.5).astype(int)
        accuracy = float(np.mean(predictions == y_te))
        test_accuracies.append(accuracy)
        
    return test_accuracies
