import numpy as np

def gan_step(real_data: list, z: list, W_g: list, b_g: list, W_d: list, b_d: list, lr: float) -> dict:
    """
    Returns updated W_g, b_g, W_d, b_d and scalar d_loss, g_loss.
    """
    # Convert inputs to float64 NumPy arrays
    X = np.array(real_data, dtype=np.float64)
    Z = np.array(z, dtype=np.float64)
    Wg = np.array(W_g, dtype=np.float64)
    bg = np.array(b_g, dtype=np.float64)
    Wd = np.array(W_d, dtype=np.float64)
    bd = np.array(b_d, dtype=np.float64)
    
    N = X.shape[0]
    
    # Helper functions
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def softplus(x):
        return np.log(1.0 + np.exp(x))
    
    # --- PHASE 1: DISCRIMINATOR STEP ---
    # 1. Generator forward pass (held constant for D)
    H = Z @ Wg.T + bg
    F = np.tanh(H)
    
    # 2. Discriminator forward pass
    a_X = X @ Wd.T + bd
    a_F = F @ Wd.T + bd
    
    # 3. Compute D loss before the update
    d_loss = np.mean(softplus(-a_X) + softplus(a_F))
    
    # 4. Compute D gradients
    # d(softplus(-x))/dx = -sigmoid(-x) = sigmoid(x) - 1
    # d(softplus(x))/dx = sigmoid(x)
    da_X = (sigmoid(a_X) - 1.0) / N
    da_F = sigmoid(a_F) / N
    
    dWd = da_X.T @ X + da_F.T @ F
    dbd = np.sum(da_X + da_F, axis=0)
    
    # 5. Update D weights
    Wd_new = Wd - lr * dWd
    bd_new = bd - lr * dbd
    
    # --- PHASE 2: GENERATOR STEP ---
    # 1. Forward pass using updated D parameters
    a_F_new = F @ Wd_new.T + bd_new
    
    # 2. Compute G loss after D update but before G update
    g_loss = np.mean(softplus(-a_F_new))
    
    # 3. Compute G gradients via backpropagation
    # dL_G / da_F_new
    da_F_new = (sigmoid(a_F_new) - 1.0) / N
    
    # dL_G / dF
    dF = da_F_new @ Wd_new
    
    # dL_G / dH (derivative of tanh is 1 - tanh^2)
    dH = dF * (1.0 - F**2)
    
    # Gradients w.r.t Generator parameters
    dWg = dH.T @ Z
    dbg = np.sum(dH, axis=0)
    
    # 4. Update G weights
    Wg_new = Wg - lr * dWg
    bg_new = bg - lr * dbg
    
    return {
        "W_g": Wg_new.tolist(),
        "b_g": bg_new.tolist(),
        "W_d": Wd_new.tolist(),
        "b_d": bd_new.tolist(),
        "d_loss": float(d_loss),
        "g_loss": float(g_loss)
    }
