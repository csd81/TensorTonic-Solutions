import numpy as np

def inception_module(x: list, W_1x1: list, b_1x1: list, W_3x3: list, b_3x3: list) -> dict:
    # Convert all inputs to float64 NumPy arrays
    x = np.array(x, dtype=np.float64)
    W_1x1 = np.array(W_1x1, dtype=np.float64)
    b_1x1 = np.array(b_1x1, dtype=np.float64)
    W_3x3 = np.array(W_3x3, dtype=np.float64)
    b_3x3 = np.array(b_3x3, dtype=np.float64)
    
    C_in, H, W = x.shape
    C1 = W_1x1.shape[0]
    C3 = W_3x3.shape[0]
    
    # 1. Branch 1x1: Pointwise cross-correlation
    # Multiplies W_1x1 (C1, C_in) with x (C_in, H, W) -> (C1, H, W)
    out_1x1 = np.tensordot(W_1x1, x, axes=([1], [0]))
    out_1x1 += b_1x1[:, np.newaxis, np.newaxis]
    out_1x1 = np.maximum(0, out_1x1)  # ReLU
    
    # 2. Branch 3x3: Zero-padded 3x3 cross-correlation (pad=1 to keep H, W)
    padded_x = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode='constant')
    out_3x3 = np.zeros((C3, H, W), dtype=np.float64)
    
    for c3 in range(C3):
        for h in range(H):
            for w in range(W):
                # Slice out the (C_in, 3, 3) patch matching the current window
                patch = padded_x[:, h:h+3, w:w+3]
                out_3x3[c3, h, w] = np.sum(patch * W_3x3[c3])
                
    out_3x3 += b_3x3[:, np.newaxis, np.newaxis]
    out_3x3 = np.maximum(0, out_3x3)  # ReLU
    
    # 3. Concatenate branches along the channel axis (axis 0)
    output = np.concatenate([out_1x1, out_3x3], axis=0)
    
    # Return dictionary with outputs rounded to 4 decimal places
    return {
        "branch_1x1": np.round(out_1x1, 4).tolist(),
        "branch_3x3": np.round(out_3x3, 4).tolist(),
        "output": np.round(output, 4).tolist()
    }
