import numpy as np

def activation_derivative(name: str, x: list) -> list:
    """
    Returns one analytical derivative for each input value.
    """
    # Convert input to a NumPy array for vectorized math operations
    x_arr = np.array(x, dtype=np.float64)
    
    if name == "sigmoid":
        # σ(x) = 1 / (1 + exp(-x))
        sig = 1.0 / (1.0 + np.exp(-x_arr))
        # σ'(x) = σ(x) * (1 - σ(x))
        deriv = sig * (1.0 - sig)
        
    elif name == "tanh":
        # tanh'(x) = 1 - tanh^2(x)
        deriv = 1.0 - np.tanh(x_arr) ** 2
        
    elif name == "relu":
        # ReLU'(x) = 1 if x > 0 else 0 (using convention ReLU'(0) = 0)
        deriv = np.where(x_arr > 0.0, 1.0, 0.0)
        
    elif name == "swish":
        # σ(x) = 1 / (1 + exp(-x))
        sig = 1.0 / (1.0 + np.exp(-x_arr))
        # Swish'(x) = σ(x) + x * σ(x) * (1 - σ(x))
        deriv = sig + x_arr * sig * (1.0 - sig)
        
    else:
        raise ValueError(f"Unknown activation function: {name}")
        
    # Convert the NumPy array back to a standard Python list of floats
    return deriv.tolist()
