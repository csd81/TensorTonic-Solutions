import numpy as np

def nesterov_momentum_step(w: list, v: list, grad: list, lr: float = 0.01, momentum: float = 0.9) -> dict:
    """
    Returns a dictionary with new_w and new_v.
    """
    w_arr = np.asarray(w, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    grad_arr = np.asarray(grad, dtype=float)
    
    new_v = momentum * v_arr + lr * grad_arr
    new_w = w_arr - new_v
    
    return {
        "new_w": new_w,
        "new_v": new_v
    }
