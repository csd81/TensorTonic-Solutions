import numpy as np

def activation_continuity_analysis(x: list) -> dict:
    """
    Returns nondifferentiable tested points for three activations.
    """
    h = np.float64(1e-7)
    tol = np.float64(1e-5)
    
    # Define the activation functions ensuring float64 precision
    def relu(v):
        return np.where(v > 0, v, np.float64(0.0))
        
    def leaky_relu(v):
        return np.where(v > 0, v, np.float64(0.01) * v)
        
    def gelu(v):
        # Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        inner = np.sqrt(2.0 / np.pi, dtype=np.float64) * (v + np.float64(0.044715) * (v ** 3))
        return np.float64(0.5) * v * (np.float64(1.0) + np.tanh(inner))

    # Containers for results
    res_relu = []
    res_leaky = []
    res_gelu = []
    
    # Evaluate each element individually
    for val in x:
        v = np.float64(val)
        
        # 1. Check ReLU
        f_left_relu = (relu(v) - relu(v - h)) / h
        f_right_relu = (relu(v + h) - relu(v)) / h
        if np.abs(f_left_relu - f_right_relu) >= tol:
            res_relu.append(val)
            
        # 2. Check Leaky ReLU
        f_left_leaky = (leaky_relu(v) - leaky_relu(v - h)) / h
        f_right_leaky = (leaky_relu(v + h) - leaky_relu(v)) / h
        if np.abs(f_left_leaky - f_right_leaky) >= tol:
            res_leaky.append(val)
            
        # 3. Check GELU
        f_left_gelu = (gelu(v) - gelu(v - h)) / h
        f_right_gelu = (gelu(v + h) - gelu(v)) / h
        if np.abs(f_left_gelu - f_right_gelu) >= tol:
            res_gelu.append(val)
            
    return {
        "relu": res_relu,
        "leaky_relu": res_leaky,
        "gelu": res_gelu
    }
