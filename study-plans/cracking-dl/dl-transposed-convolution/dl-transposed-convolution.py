import numpy as np

def transposed_conv2d(input: list, filters: list, bias=None, stride: int = 1, padding: int = 0) -> list:
    """
    Returns the channel-first transposed-convolution output.
    """
    X = np.array(input, dtype=float)
    F = np.array(filters, dtype=float)
    
    C_in, H, W = X.shape
    _, C_out, K_H, K_W = F.shape
    
    # 1. Compute full spatial sizes before cropping
    H_f = (H - 1) * stride + K_H
    W_f = (W - 1) * stride + K_W
    
    # 2. Initialize the accumulation grid
    out_full = np.zeros((C_out, H_f, W_f), dtype=float)
    
    # 3. Scatter each input value scaled by its filter weights
    for c in range(C_in):
        for i in range(H):
            for j in range(W):
                val = X[c, i, j]
                # Start index for scattering
                r_start = i * stride
                c_start = j * stride
                
                # Accumulate the weighted filter contribution
                out_full[:, r_start : r_start + K_H, c_start : c_start + K_W] += val * F[c]
                
    # 4. Crop padding cells from every edge
    # Avoid out-of-bounds slicing by correctly identifying the end index
    h_end = H_f - padding
    w_end = W_f - padding
    out_cropped = out_full[:, padding:h_end, padding:w_end]
    
    # 5. Add bias per output channel if available
    if bias is not None:
        B = np.array(bias, dtype=float)
        out_cropped += B[:, np.newaxis, np.newaxis]
        
    # 6. Round to 4 decimal places and return as nested lists
    return np.round(out_cropped, 4).tolist()
