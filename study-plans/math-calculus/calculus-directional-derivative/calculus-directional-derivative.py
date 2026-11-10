import numpy as np

def directional_derivative_analysis(w: list, v: list) -> dict:
    """
    Returns gradient and directional steepest-descent data.
    """
    w1, w2 = w[0], w[1]
    
    # 1. Compute the analytical gradient: L = w1^2 + 3*w2^2
    # dL/dw1 = 2*w1, dL/dw2 = 6*w2
    grad_w1 = 2.0 * w1
    grad_w2 = 6.0 * w2
    gradient = [grad_w1, grad_w2]
    
    # 2. Compute the directional derivative: D_v L = grad · v
    directional_derivative = float(grad_w1 * v[0] + grad_w2 * v[1])
    
    # Calculate gradient magnitude
    grad_norm = np.sqrt(grad_w1**2 + grad_w2**2)
    
    # 3. Handle zero gradient case, otherwise normalize the negative gradient
    if grad_norm == 0:
        steepest_descent_direction = [0.0, 0.0]
        max_descent_rate = 0.0
    else:
        steepest_descent_direction = [-grad_w1 / grad_norm, -grad_w2 / grad_norm]
        # Maximum rate of descent is the negative magnitude of the gradient
        max_descent_rate = float(-grad_norm)
        
    return {
        "gradient": gradient,
        "directional_derivative": directional_derivative,
        "steepest_descent_direction": steepest_descent_direction,
        "max_descent_rate": max_descent_rate
    }
