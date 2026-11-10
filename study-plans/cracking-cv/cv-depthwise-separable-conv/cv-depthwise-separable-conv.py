import numpy as np

def depthwise_separable_conv(
    image: list, 
    depthwise_weight: list, 
    pointwise_weight: list, 
    bias: list, 
    stride: int, 
    padding: int
) -> list:
    """
    Returns the depthwise-separable result as a channel-first nested list.
    """
    # Convert inputs to float64 numpy arrays for precise arithmetic
    img = np.array(image, dtype=np.float64)                # Shape: (Cin, H, W)
    dw = np.array(depthwise_weight, dtype=np.float64)      # Shape: (Cin, 1, Kh, Kw)
    pw = np.array(pointwise_weight, dtype=np.float64)      # Shape: (Cout, Cin, 1, 1)
    b = np.array(bias, dtype=np.float64)                  # Shape: (Cout,)
    
    Cin, H, W = img.shape
    _, _, Kh, Kw = dw.shape
    Cout, _, _, _ = pw.shape
    
    # 1. Apply spatial padding to the input image
    if padding > 0:
        padded_img = np.pad(
            img, 
            ((0, 0), (padding, padding), (padding, padding)), 
            mode='constant', 
            constant_values=0
        )
    else:
        padded_img = img
        
    # Calculate exact output dimensions
    Hout = (H + 2 * padding - Kh) // stride + 1
    Wout = (W + 2 * padding - Kw) // stride + 1
    
    # 2. Depthwise Convolution: Process each channel independently
    D = np.zeros((Cin, Hout, Wout), dtype=np.float64)
    for c in range(Cin):
        kernel = dw[c, 0]  # Spatial filter for channel c
        for i in range(Hout):
            for j in range(Wout):
                r_start = i * stride
                c_start = j * stride
                window = padded_img[c, r_start:r_start+Kh, c_start:c_start+Kw]
                D[c, i, j] = np.sum(window * kernel)
                
    # 3. Pointwise Convolution: 1x1 projection across all channels + bias
    out = np.zeros((Cout, Hout, Wout), dtype=np.float64)
    for o in range(Cout):
        for i in range(Hout):
            for j in range(Wout):
                val = b[o]
                for c in range(Cin):
                    val += pw[o, c, 0, 0] * D[c, i, j]
                out[o, i, j] = val
                
    # Recursive helper to build the nested list cleanly rounded to 4 decimal places
    def to_nested_list(arr):
        if arr.ndim == 1:
            return [round(x, 4) for x in arr.tolist()]
        return [to_nested_list(sub) for sub in arr]
        
    return to_nested_list(out)
