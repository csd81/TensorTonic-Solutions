import numpy as np

def conv2d(input: list, filters: list, bias=None, padding: int = 0, stride: int = 1) -> list:
    """
    Returns the channel-first cross-correlation output.
    """
    # Convert inputs to NumPy arrays with float64 precision
    X = np.array(input, dtype=np.float64)
    F = np.array(filters, dtype=np.float64)
    
    C_in, H, W = X.shape
    C_out, _, KH, KW = F.shape
    
    # Handle optional bias
    if bias is None:
        B = np.zeros(C_out, dtype=np.float64)
    else:
        B = np.array(bias, dtype=np.float64)
        
    # Apply zero padding to spatial dimensions (Height and Width)
    if padding > 0:
        X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    else:
        X_padded = X
        
    _, H_p, W_p = X_padded.shape
    
    # Calculate floor-based valid output dimensions
    H_out = int((H_p - KH) // stride) + 1
    W_out = int((W_p - KW) // stride) + 1
    
    # Initialize output tensor
    Y = np.zeros((C_out, H_out, W_out), dtype=np.float64)
    
    # Perform cross-correlation
    for o in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                h_end = h_start + KH
                w_start = j * stride
                w_end = w_start + KW
                
                # Extract the local patch across all input channels
                patch = X_padded[:, h_start:h_end, w_start:w_end]
                
                # Element-wise multiply, sum across all dimensions, and add bias
                Y[o, i, j] = np.sum(patch * F[o]) + B[o]
                
    # Round to 4 decimal places and return as a nested list
    return np.round(Y, 4).tolist()
