import numpy as np

def nadam_step(w: list, m: list, v: list, grad: list, lr: float = 0.002, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    # Convert inputs to float64 NumPy arrays for vectorized math and precision
    w_arr = np.asarray(w, dtype=np.float64)
    m_arr = np.asarray(m, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    g_arr = np.asarray(grad, dtype=np.float64)

    # 1. Update biased first and second moment estimates
    new_m = beta1 * m_arr + (1.0 - beta1) * g_arr
    new_v = beta2 * v_arr + (1.0 - beta2) * (g_arr ** 2)

    # 2. Compute Nesterov-adjusted first moment
    m_tilde = beta1 * new_m + (1.0 - beta1) * g_arr

    # 3. Update parameters
    new_w = w_arr - lr * m_tilde / (np.sqrt(new_v) + eps)

    return {
        "new_w": new_w,
        "new_m": new_m,
        "new_v": new_v
    }
