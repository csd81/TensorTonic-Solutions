import numpy as np

def lenet_forward(x: list, W_conv: list, b_conv: list, W_fc1: list, b_fc1: list, W_fc2: list, b_fc2: list) -> dict:
    """
    Returns convolution, pooling, hidden, and logit outputs.
    """
    # 1. Convert all inputs to float64 NumPy arrays
    x = np.array(x, dtype=np.float64)
    W_conv = np.array(W_conv, dtype=np.float64)
    b_conv = np.array(b_conv, dtype=np.float64)
    W_fc1 = np.array(W_fc1, dtype=np.float64)
    b_fc1 = np.array(b_fc1, dtype=np.float64)
    W_fc2 = np.array(W_fc2, dtype=np.float64)
    b_fc2 = np.array(b_fc2, dtype=np.float64)
    
    C_out, C_in, k_h, k_w = W_conv.shape
    _, H, W = x.shape
    
    # 2. Convolution (Valid Cross-Correlation) + Bias + ReLU
    out_h = H - k_h + 1
    out_w = W - k_w + 1
    conv_features = np.zeros((C_out, out_h, out_w), dtype=np.float64)
    
    for c_o in range(C_out):
        for i in range(out_h):
            for j in range(out_w):
                # Extract patch across all input channels
                patch = x[:, i:i+k_h, j:j+k_w]
                # Element-wise multiplication and summation over all channels and space
                conv_features[c_o, i, j] = np.sum(patch * W_conv[c_o]) + b_conv[c_o]
    
    # Apply ReLU activation
    conv_out = np.maximum(0, conv_features)
    
    # 3. Max Pooling (2x2 non-overlapping)
    pool_h = out_h // 2
    pool_w = out_w // 2
    pool_out = np.zeros((C_out, pool_h, pool_w), dtype=np.float64)
    
    for c_o in range(C_out):
        for i in range(pool_h):
            for j in range(pool_w):
                # Extra steps of size 2 (non-overlapping stride)
                r, c = i * 2, j * 2
                pool_out[c_o, i, j] = np.max(conv_out[c_o, r:r+2, c:c+2])
                
    # 4. Flattening (Row-major order / C-contiguous)
    flat_out = pool_out.flatten()
    
    # 5. Fully Connected Layer 1 (Affine + ReLU)
    fc1_linear = np.dot(W_fc1, flat_out) + b_fc1
    fc1_out = np.maximum(0, fc1_linear)
    
    # 6. Fully Connected Layer 2 (Linear Output)
    logits = np.dot(W_fc2, fc1_out) + b_fc2
    
    # Return dictionary with everything rounded to 4 decimal places
    return {
        "conv_out": np.round(conv_out, 4).tolist(),
        "pool_out": np.round(pool_out, 4).tolist(),
        "fc1_out": np.round(fc1_out, 4).tolist(),
        "logits": np.round(logits, 4).tolist()
    }
