import numpy as np

def subgradient_analysis(x_points: list, w_init: float, lr: float, n_iters: int) -> dict:
    """
    Returns selected subgradients and the descent trajectory.
    """
    # Convert points list to a NumPy array for vectorized subgradient calculations
    x_arr = np.array(x_points, dtype=np.float64)
    
    # 1. Absolute value subgradient using the convention: sign(x) with sign(0) = 0
    abs_subgrad = np.where(x_arr > 0.0, 1.0, np.where(x_arr < 0.0, -1.0, 0.0))
    
    # 2. ReLU subgradient using the convention: 1 if x > 0 else 0
    relu_subgrad = np.where(x_arr > 0.0, 1.0, 0.0)
    
    # 3. Simulate subgradient descent optimization loop
    w_trajectory = [float(w_init)]
    w_current = float(w_init)
    
    for _ in range(n_iters):
        # Compute subgradient of absolute component using sign convention at (w - 3)
        diff = w_current - 3.0
        sign_diff = 1.0 if diff > 0.0 else (-1.0 if diff < 0.0 else 0.0)
        
        # Total subgradient: g_t = sign(w_t - 3) + w_t
        g_t = sign_diff + w_current
        
        # Optimization update step
        w_current = w_current - lr * g_t
        w_trajectory.append(float(w_current))
        
    return {
        "abs_subgrad": abs_subgrad.tolist(),
        "relu_subgrad": relu_subgrad.tolist(),
        "w_trajectory": w_trajectory,
        "w_final": w_current
    }
