import numpy as np

def vgg_block(x: list, W1: list, b1: list, W2: list, b2: list) -> dict:
    """
    Applies two stages of 3x3 valid cross-correlation followed by ReLU,
    and a final 2x2 max pooling layer with stride 2.
    """
    # Convert inputs to float64 NumPy arrays
    x = np.array(x, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    b1 = np.array(b1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    b2 = np.array(b2, dtype=np.float64)
    
    # --- Conv 1 Stage ---
    C1, C_in, k_h, k_w = W1.shape
    H, W = x.shape[1], x.shape[2]
    
    H_out1 = H - k_h + 1
    W_out1 = W - k_w + 1
    
    conv1 = np.zeros((C1, H_out1, W_out1), dtype=np.float64)
    for c1 in range(C1):
        for h in range(H_out1):
            for w in range(W_out1):
                patch = x[:, h:h+k_h, w:w+k_w]
                conv1[c1, h, w] = np.sum(patch * W1[c1]) + b1[c1]
                
    # Apply first ReLU activation
    conv1 = np.maximum(0, conv1)
    
    # --- Conv 2 Stage ---
    C2, _, k_h2, k_w2 = W2.shape
    H_out2 = H_out1 - k_h2 + 1
    W_out2 = W_out1 - k_w2 + 1
    
    conv2 = np.zeros((C2, H_out2, W_out2), dtype=np.float64)
    for c2 in range(C2):
        for h in range(H_out2):
            for w in range(W_out2):
                patch = conv1[:, h:h+k_h2, w:w+k_w2]
                conv2[c2, h, w] = np.sum(patch * W2[c2]) + b2[c2]
                
    # Apply second ReLU activation
    conv2 = np.maximum(0, conv2)
    
    # --- Max Pooling Stage ---
    H_p = H_out2 // 2
    W_p = W_out2 // 2
    pool = np.zeros((C2, H_p, W_p), dtype=np.float64)
    
    for c2 in range(C2):
        for h in range(H_p):
            for w in range(W_p):
                patch = conv2[c2, h*2:h*2+2, w*2:w*2+2]
                pool[c2, h, w] = np.max(patch)
                
    # Round results to 4 decimal places and convert back to nested lists
    return {
        "conv1": np.round(conv1, 4).tolist(),
        "conv2": np.round(conv2, 4).tolist(),
        "pool": np.round(pool, 4).tolist()
    }
