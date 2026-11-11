import numpy as np

def depthwise_separable_conv(x: list, W_dw: list, b_dw: list, W_pw: list, b_pw: list) -> dict:
    """
    Computes depthwise separable convolution using valid 3x3 depthwise filters 
    followed by a 1x1 pointwise mixing layer, both with ReLU activation.
    """
    # Convert inputs to float64 arrays
    x = np.array(x, dtype=np.float64)
    W_dw = np.array(W_dw, dtype=np.float64)
    b_dw = np.array(b_dw, dtype=np.float64)
    W_pw = np.array(W_pw, dtype=np.float64)
    b_pw = np.array(b_pw, dtype=np.float64)
    
    C_in, H, W = x.shape
    H_out, W_out = H - 2, W - 2
    
    # 1. Depthwise Convolution Stage
    D = np.zeros((C_in, H_out, W_out), dtype=np.float64)
    for c in range(C_in):
        for i in range(H_out):
            for j in range(W_out):
                window = x[c, i:i+3, j:j+3]
                D[c, i, j] = np.sum(window * W_dw[c]) + b_dw[c]
    
    # Apply ReLU to Depthwise output
    D = np.maximum(0.0, D)
    
    # 2. Pointwise Convolution Stage (1x1 Mixing)
    # Using np.tensordot to mix channels efficiently across spatial locations
    P = np.tensordot(W_pw, D, axes=(1, 0)) + b_pw[:, None, None]
    
    # Apply ReLU to Pointwise output
    P = np.maximum(0.0, P)
    
    # Round results to 4 decimal places and convert back to nested lists
    return {
        "depthwise": np.round(D, 4).tolist(),
        "pointwise": np.round(P, 4).tolist()
    }
