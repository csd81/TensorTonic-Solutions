import numpy as np

def adamw_step(
    w: list, 
    m: list, 
    v: list, 
    grad: list, 
    lr: float = 0.001, 
    beta1: float = 0.9, 
    beta2: float = 0.999, 
    weight_decay: float = 0.01, 
    eps: float = 1e-8
) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    w = np.asarray(w, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    g = np.asarray(grad, dtype=np.float64)

    new_m = beta1 * m + (1.0 - beta1) * g
    new_v = beta2 * v + (1.0 - beta2) * (g ** 2)

    adaptive_step = lr * new_m / (np.sqrt(new_v) + eps)
    decay_step = lr * weight_decay * w
    new_w = w - adaptive_step - decay_step

    return {
        "new_w": new_w,
        "new_m": new_m,
        "new_v": new_v
    }
