import numpy as np

def lbfgs(X: list, y: list, lr_gd: float, m: int, n_iters: int) -> dict:
    """
    Returns gradient-descent and L-BFGS MSE curves in a dictionary.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n_samples, n_features = X.shape
    
    def get_mse(w):
        return np.mean((X @ w - y) ** 2)
        
    def get_grad(w):
        return (2.0 / n_samples) * X.T @ (X @ w - y)
        
    # --- Gradient Descent ---
    w_gd = np.zeros(n_features, dtype=np.float64)
    gd_losses = [float(get_mse(w_gd))]
    for _ in range(n_iters):
        w_gd = w_gd - lr_gd * get_grad(w_gd)
        gd_losses.append(float(get_mse(w_gd)))
        
    # --- L-BFGS ---
    w_lb = np.zeros(n_features, dtype=np.float64)
    lbfgs_losses = [float(get_mse(w_lb))]
    
    s_history = []
    z_history = []
    
    for _ in range(n_iters):
        g_k = get_grad(w_lb)
        
        # Two-loop recursion
        q = g_k.copy()
        alphas = []
        
        k = len(s_history)
        # 1. Backward pass
        for i in reversed(range(k)):
            s_i = s_history[i]
            z_i = z_history[i]
            rho_i = 1.0 / (np.dot(s_i, z_i) + 1e-10)
            alpha_i = rho_i * np.dot(s_i, q)
            alphas.append(alpha_i)
            q -= alpha_i * z_i
            
        alphas.reverse()  # Match indices for the forward pass
        
        # Initial Hessian scaling
        if k == 0:
            gamma = 1.0
        else:
            s_last = s_history[-1]
            z_last = z_history[-1]
            gamma = np.dot(s_last, z_last) / (np.dot(z_last, z_last) + 1e-10)
            
        r = gamma * q
        
        # 2. Forward pass
        for i in range(k):
            s_i = s_history[i]
            z_i = z_history[i]
            rho_i = 1.0 / (np.dot(s_i, z_i) + 1e-10)
            beta_i = rho_i * np.dot(z_i, r)
            r += s_i * (alphas[i] - beta_i)
            
        d = -r
        
        # Armijo backtracking line search
        alpha_step = 1.0
        f_k = get_mse(w_lb)
        c1 = 0.0001
        
        for _ in range(21):  # up to 20 halvings means 21 evaluations max
            w_next = w_lb + alpha_step * d
            if get_mse(w_next) <= f_k + c1 * alpha_step * np.dot(g_k, d):
                break
            alpha_step *= 0.5
            
        w_next = w_lb + alpha_step * d
        g_next = get_grad(w_next)
        
        # Record history element
        s_k = w_next - w_lb
        z_k = g_next - g_k
        
        s_history.append(s_k)
        z_history.append(z_k)
        if len(s_history) > m:
            s_history.pop(0)
            z_history.pop(0)
            
        w_lb = w_next
        lbfgs_losses.append(float(get_mse(w_lb)))
        
    return {"gd_losses": gd_losses, "lbfgs_losses": lbfgs_losses}
