import numpy as np

def adadelta_step(w: list, grad: list, E_grad_sq: list, E_update_sq: list, rho: float = 0.9, eps: float = 1e-6) -> dict:
    """
    Returns a dictionary with new_w, new_E_grad_sq, and new_E_update_sq.
    """
    # Convert inputs to float64 NumPy arrays for element-wise math and precision
    w = np.asarray(w, dtype=np.float64)
    g = np.asarray(grad, dtype=np.float64)
    E_g2 = np.asarray(E_grad_sq, dtype=np.float64)
    E_dw2 = np.asarray(E_update_sq, dtype=np.float64)

    # 1. Update running squared-gradient average
    new_E_grad_sq = rho * E_g2 + (1.0 - rho) * (g ** 2)

    # 2. Compute the parameter change using previous E_update_sq and current E_grad_sq
    rms_update = np.sqrt(E_dw2 + eps)
    rms_grad = np.sqrt(new_E_grad_sq + eps)
    delta_w = -(rms_update / rms_grad) * g

    # 3. Update running squared-change average
    new_E_update_sq = rho * E_dw2 + (1.0 - rho) * (delta_w ** 2)

    # 4. Update the parameters
    new_w = w + delta_w

    return {
        "new_w": new_w,
        "new_E_grad_sq": new_E_grad_sq,
        "new_E_update_sq": new_E_update_sq,
    }
