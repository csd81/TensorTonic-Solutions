import numpy as np

def conv2d(image: list, weight: list, bias: list, stride: int, padding: int) -> list:
    """
    Returns the channel-first convolution result as a nested list.
    """
    # Convert inputs to float64 NumPy arrays as required
    X = np.array(image, dtype=np.float64)
    W = np.array(weight, dtype=np.float64)
    B = np.array(bias, dtype=np.float64)
    
    C_in, H, W_in = X.shape
    C_out, _, K_h, K_w = W.shape
    
    # Apply zero padding to spatial dimensions (H, W_in)
    if padding > 0:
        X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    else:
        X_padded = X
        
    _, H_pad, W_pad = X_padded.shape
    
    # Calculate output dimensions
    H_out = int((H + 2 * padding - K_h) // stride) + 1
    W_out = int((W_in + 2 * padding - K_w) // stride) + 1
    
    # Initialize output array
    out = np.zeros((C_out, H_out, W_out), dtype=np.float64)
    
    # Perform cross-correlation sliding window
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + K_h
            w_start = j * stride
            w_end = w_start + K_w
            
            # Extract current sliding window across all input channels
            # Shape: (C_in, K_h, K_w)
            window = X_padded[:, h_start:h_end, w_start:w_end]
            
            # Multiply element-wise with kernels and sum over C_in, K_h, K_w for each C_out
            # Adding the channel bias dynamically
            out[:, i, j] = np.sum(W * window, axis=(1, 2, 3)) + B
            
    # Round to 4 decimal places and convert back to a nested Python list
    return np.round(out, 4).tolist()
