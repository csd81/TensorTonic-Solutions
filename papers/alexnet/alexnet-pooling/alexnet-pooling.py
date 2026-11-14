import numpy as np

def max_pool2d(x: np.ndarray, kernel_size: int, stride: int) -> np.ndarray:
    """
    Returns the float64 NHWC max-pooled tensor.
    """
    B, H, W, C = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    
    # Initialize output array as float64
    out = np.zeros((B, H_out, W_out, C), dtype=np.float64)
    
    # Cast input to float64 to ensure the result is float64
    x = x.astype(np.float64)
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + kernel_size
            w_start = j * stride
            w_end = w_start + kernel_size
            
            # Extract window of shape (B, kernel_size, kernel_size, C)
            window = x[:, h_start:h_end, w_start:w_end, :]
            
            # Take max along spatial dimensions (axes 1 and 2 of the window)
            out[:, i, j, :] = np.max(window, axis=(1, 2))
            
    return out
