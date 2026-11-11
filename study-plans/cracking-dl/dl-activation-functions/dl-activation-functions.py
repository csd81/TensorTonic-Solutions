import numpy as np

def activation_functions(x: float, activation: str) -> list:
    """
    Returns the activation value and analytical derivative, 
    rounded to 4 decimal places.
    """
    out = 0.0
    deriv = 0.0
    
    if activation == "relu":
        if x > 0:
            out = x
            deriv = 1.0
        else:
            out = 0.0
            deriv = 0.0  # Defined as 0 at zero per requirements
            
    elif activation == "leaky_relu":
        if x > 0:
            out = x
            deriv = 1.0
        else:
            out = 0.01 * x
            deriv = 0.01  # Defined as 0.01 at zero per requirements
            
    elif activation == "sigmoid":
        # Standard sigmoid: 1 / (1 + exp(-x))
        out = 1.0 / (1.0 + np.exp(-x))
        # Derivative: sigmoid * (1 - sigmoid)
        deriv = out * (1.0 - out)
        
    elif activation == "tanh":
        out = np.tanh(x)
        # Derivative: 1 - tanh^2
        deriv = 1.0 - out ** 2
        
    elif activation == "gelu":
        # Constants for the tanh approximation
        c1 = np.sqrt(2.0 / np.pi)
        c2 = 0.044715
        
        # Inner terms
        inner_poly = x + c2 * (x ** 3)
        tanh_arg = c1 * inner_poly
        t = np.tanh(tanh_arg)
        
        # Output calculation
        out = 0.5 * x * (1.0 + t)
        
        # Chain rule derivative calculation
        # d/dx [inner_poly] = 1 + 3 * 0.044715 * x^2
        d_inner_poly = 1.0 + 3.0 * c2 * (x ** 2)
        # d/dx [tanh_arg] = c1 * d_inner_poly
        d_tanh_arg = c1 * d_inner_poly
        # d/dx [t] = (1 - t^2) * d_tanh_arg
        dt = (1.0 - t ** 2) * d_tanh_arg
        
        # Derivative using product rule on: 0.5 * x * (1 + t)
        deriv = 0.5 * (1.0 + t) + 0.5 * x * dt
        
    elif activation == "swish":
        # Swish(x) = x * sigmoid(x)
        sig = 1.0 / (1.0 + np.exp(-x))
        out = x * sig
        # Derivative: Swish(x) + sigmoid(x) * (1 - Swish(x))
        deriv = out + sig * (1.0 - out)
        
    return [round(out, 4), round(deriv, 4)]
