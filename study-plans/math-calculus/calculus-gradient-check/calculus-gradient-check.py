import numpy as np

def gradient_check(f_name: str, x: float, analytical_deriv: float) -> dict:
    """
    Returns finite-difference errors and the best h for each method.
    """
    # 1. Define the internal function registry
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
        
    funcs = {
        "sin": np.sin,
        "exp": np.exp,
        "sigmoid": sigmoid,
        "x_cubed": lambda z: z ** 3
    }
    
    f = funcs[f_name]
    
    # 2. Setup the exact array of 8 h values from 10^-1 down to 10^-8
    h_values = [10**(-i) for i in range(1, 9)]
    
    forward_errors = []
    backward_errors = []
    central_errors = []
    
    # 3. Calculate numerical derivatives and their absolute errors for each step size
    for h in h_values:
        f_x = f(x)
        f_plus = f(x + h)
        f_minus = f(x - h)
        
        g_forward = (f_plus - f_x) / h
        g_backward = (f_x - f_minus) / h
        g_central = (f_plus - f_minus) / (2.0 * h)
        
        forward_errors.append(abs(g_forward - analytical_deriv))
        backward_errors.append(abs(g_backward - analytical_deriv))
        central_errors.append(abs(g_central - analytical_deriv))
        
    # 4. Find the optimal h for each layout (first index where minimum occurs)
    opt_fwd_idx = np.argmin(forward_errors)
    opt_bwd_idx = np.argmin(backward_errors)
    opt_ctr_idx = np.argmin(central_errors)
    
    return {
        "h_values": h_values,
        "forward_errors": forward_errors,
        "backward_errors": backward_errors,
        "central_errors": central_errors,
        "optimal_h_forward": h_values[opt_fwd_idx],
        "optimal_h_backward": h_values[opt_bwd_idx],
        "optimal_h_central": h_values[opt_ctr_idx]
    }
