import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Convert inputs to NumPy arrays for efficient elementwise operations
    theta = np.array(param, dtype=np.float64)
    g = np.array(grad, dtype=np.float64)
    m_prev = np.array(m, dtype=np.float64)
    v_prev = np.array(v, dtype=np.float64)
    
    # Step 1: Update First Moment
    m_new = beta1 * m_prev + (1 - beta1) * g
    
    # Step 2: Update Second Moment
    v_new = beta2 * v_prev + (1 - beta2) * (g ** 2)
    
    # Step 3: Bias Correction
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    
    # Step 4: Parameter Update
    param_new = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return param_new, m_new, v_new
