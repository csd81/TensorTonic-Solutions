import numpy as np

def taylor_approximation(w0: float, n_points: int = 100) -> dict:
    """
    Returns the sampled loss and its first two Taylor approximations.
    """
    # Generate n_points evenly spaced values from w0 - 2 to w0 + 2
    w_values = np.linspace(w0 - 2, w0 + 2, n_points)
    
    # Analytical calculations at the expansion point w0
    L_at_w0 = np.sin(w0) + 0.1 * (w0 ** 2)
    dL_at_w0 = np.cos(w0) + 0.2 * w0
    d2L_at_w0 = -np.sin(w0) + 0.2
    
    # Evaluate the true loss across all w values
    true_loss = np.sin(w_values) + 0.1 * (w_values ** 2)
    
    # Compute first-order (linear) approximation: T1(w) = L(w0) + L'(w0)*(w - w0)
    linear_approx = L_at_w0 + dL_at_w0 * (w_values - w0)
    
    # Compute second-order (quadratic) approximation: T2(w) = T1(w) + 0.5 * L''(w0)*(w - w0)^2
    quadratic_approx = linear_approx + 0.5 * d2L_at_w0 * ((w_values - w0) ** 2)
    
    return {
        "L_at_w0": float(L_at_w0),
        "dL_at_w0": float(dL_at_w0),
        "d2L_at_w0": float(d2L_at_w0),
        "linear_approx": [float(x) for x in linear_approx],
        "quadratic_approx": [float(x) for x in quadratic_approx],
        "true_loss": [float(x) for x in true_loss],
        "w_values": [float(x) for x in w_values]
    }
