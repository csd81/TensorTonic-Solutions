import numpy as np

def resnet_block(x: list, W1: list, b1: list, W2: list, b2: list) -> dict:
    """
    Returns the two convolution results and residual output.
    """
    # Convert inputs to float64 NumPy arrays
    X_arr = np.array(x, dtype=np.float64)
    W1_arr = np.array(W1, dtype=np.float64)
    b1_arr = np.array(b1, dtype=np.float64)
    W2_arr = np.array(W2, dtype=np.float64)
    b2_arr = np.array(b2, dtype=np.float64)
    
    def conv2d(X, W, b):
        C_out, C_in, kH, kW = W.shape
        C_in_X, H, W_X = X.shape
        
        # Add 1 zero-padding cell around both spatial axes
        X_pad = np.pad(X, ((0, 0), (1, 1), (1, 1)), mode='constant')
        out = np.zeros((C_out, H, W_X), dtype=np.float64)
        
        # Perform standard cross-correlation 2D convolution
        for c_o in range(C_out):
            for h in range(H):
                for w in range(W_X):
                    val = b[c_o]
                    for c_i in range(C_in):
                        val += np.sum(X_pad[c_i, h:h+kH, w:w+kW] * W[c_o, c_i])
                    out[c_o, h, w] = val
        return out

    # 1. First Conv + ReLU
    Z1 = conv2d(X_arr, W1_arr, b1_arr)
    A1 = np.maximum(0.0, Z1)
    
    # 2. Second Conv
    Z2 = conv2d(A1, W2_arr, b2_arr)
    
    # 3. Skip Connection + final ReLU
    Y = np.maximum(0.0, Z2 + X_arr)
    
    # Format and round to 4 decimal places
    return {
        "conv1": np.round(A1, 4).tolist(),
        "conv2": np.round(Z2, 4).tolist(),
        "output": np.round(Y, 4).tolist()
    }
