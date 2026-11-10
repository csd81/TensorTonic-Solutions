import numpy as np

def optimizer_showdown(X: list, y: list, lr: float, beta1: float, beta2: float, epsilon: float, n_epochs: int, batch_size: int) -> dict:
    """
    Compares plain SGD, classical momentum, and Adam on mini-batch logistic regression,
    returning full-dataset binary cross-entropy curves in a dictionary.
    """
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)
    n_samples, n_features = X_arr.shape
    
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
        
    def compute_bce(w, b):
        p = sigmoid(X_arr @ w + b)
        p = np.clip(p, 1e-15, 1.0 - 1e-15)
        return -np.mean(y_arr * np.log(p) + (1.0 - y_arr) * np.log(1.0 - p))

    # 1. Plain SGD
    sgd_losses = []
    w_sgd = np.zeros(n_features)
    b_sgd = 0.0
    for epoch in range(n_epochs):
        for i in range(0, n_samples, batch_size):
            X_b = X_arr[i:i+batch_size]
            y_b = y_arr[i:i+batch_size]
            r = len(y_b)
            p = sigmoid(X_b @ w_sgd + b_sgd)
            
            gw = (X_b.T @ (p - y_b)) / r
            gb = np.mean(p - y_b)
            
            w_sgd -= lr * gw
            b_sgd -= lr * gb
        sgd_losses.append(float(compute_bce(w_sgd, b_sgd)))

    # 2. Classical Momentum
    momentum_losses = []
    w_mom = np.zeros(n_features)
    b_mom = 0.0
    vw = np.zeros(n_features)
    vb = 0.0
    for epoch in range(n_epochs):
        for i in range(0, n_samples, batch_size):
            X_b = X_arr[i:i+batch_size]
            y_b = y_arr[i:i+batch_size]
            r = len(y_b)
            p = sigmoid(X_b @ w_mom + b_mom)
            
            gw = (X_b.T @ (p - y_b)) / r
            gb = np.mean(p - y_b)
            
            vw = beta1 * vw + gw
            vb = beta1 * vb + gb
            
            w_mom -= lr * vw
            b_mom -= lr * vb
        momentum_losses.append(float(compute_bce(w_mom, b_mom)))

    # 3. Adam Optimizer
    adam_losses = []
    w_adam = np.zeros(n_features)
    b_adam = 0.0
    mw = np.zeros(n_features)
    mb = 0.0
    v_adam_w = np.zeros(n_features)
    v_adam_b = 0.0
    t = 0
    for epoch in range(n_epochs):
        for i in range(0, n_samples, batch_size):
            X_b = X_arr[i:i+batch_size]
            y_b = y_arr[i:i+batch_size]
            r = len(y_b)
            p = sigmoid(X_b @ w_adam + b_adam)
            
            gw = (X_b.T @ (p - y_b)) / r
            gb = np.mean(p - y_b)
            
            t += 1
            mw = beta1 * mw + (1.0 - beta1) * gw
            mb = beta1 * mb + (1.0 - beta1) * gb
            v_adam_w = beta2 * v_adam_w + (1.0 - beta2) * (gw ** 2)
            v_adam_b = beta2 * v_adam_b + (1.0 - beta2) * (gb ** 2)
            
            mw_hat = mw / (1.0 - beta1 ** t)
            mb_hat = mb / (1.0 - beta1 ** t)
            v_adam_w_hat = v_adam_w / (1.0 - beta2 ** t)
            v_adam_b_hat = v_adam_b / (1.0 - beta2 ** t)
            
            w_adam -= lr * mw_hat / (np.sqrt(v_adam_w_hat) + epsilon)
            b_adam -= lr * mb_hat / (np.sqrt(v_adam_b_hat) + epsilon)
        adam_losses.append(float(compute_bce(w_adam, b_adam)))

    return {
        "sgd_losses": sgd_losses,
        "momentum_losses": momentum_losses,
        "adam_losses": adam_losses
    }
