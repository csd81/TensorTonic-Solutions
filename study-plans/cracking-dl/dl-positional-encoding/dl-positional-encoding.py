import numpy as np

def rope(x: list, mode: str = 'forward', d_output: list | None = None) -> dict:
    """
    Computes Rotary Position Embedding (RoPE) forward pass and optional 
    backward gradients, returning results rounded to 4 decimal places.
    """
    x_arr = np.array(x, dtype=np.float64)
    T, D = x_arr.shape
    
    # Generate sequence positions (t) and pair indices (i)
    t = np.arange(T, dtype=np.float64).reshape(T, 1)
    i = np.arange(D // 2, dtype=np.float64).reshape(1, D // 2)
    
    # Calculate frequencies and rotation angles: theta = t / (10000 ** (2i / D))
    freqs = 10000 ** (2 * i / D)
    theta = t / freqs
    
    cos_pe = np.cos(theta)
    sin_pe = np.sin(theta)
    
    # Separate adjacent pairs into even and odd indices
    x_even = x_arr[:, 0::2]
    x_odd = x_arr[:, 1::2]
    
    # Apply rotation transformation for the forward pass
    rotated = np.zeros_like(x_arr)
    rotated[:, 0::2] = x_even * cos_pe - x_odd * sin_pe
    rotated[:, 1::2] = x_even * sin_pe + x_odd * cos_pe
    
    # Construct output dictionary with rounded lists
    res = {
        'rotated': np.round(rotated, 4).tolist(),
        'cos_pe': np.round(cos_pe, 4).tolist(),
        'sin_pe': np.round(sin_pe, 4).tolist()
    }
    
    # Process backward pass gradient calculations if required
    if mode == 'backward' and d_output is not None:
        d_out_arr = np.array(d_output, dtype=np.float64)
        dx = np.zeros_like(x_arr)
        
        dy_even = d_out_arr[:, 0::2]
        dy_odd = d_out_arr[:, 1::2]
        
        # Calculate gradients with respect to input x using rotation matrix transpose
        dx[:, 0::2] = dy_even * cos_pe + dy_odd * sin_pe
        dx[:, 1::2] = -dy_even * sin_pe + dy_odd * cos_pe
        
        res['dx'] = np.round(dx, 4).tolist()
        
    return res
