import numpy as np

def _conv2d_relu(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    N, H, W, _ = x.shape
    K_h, K_w, _, C_out = kernel.shape
    
    pad_h = K_h - 1
    pad_w = K_w - 1
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    x_padded = np.pad(
        x,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )
    
    out = np.zeros((N, H, W, C_out), dtype=np.float64)
    for h in range(H):
        for w in range(W):
            patch = x_padded[:, h : h + K_h, w : w + K_w, :]
            out[:, h, w, :] = np.tensordot(patch, kernel, axes=((1, 2, 3), (0, 1, 2))) + bias
            
    return np.maximum(0.0, out)

def _max_pool2d(x: np.ndarray) -> np.ndarray:
    N, H, W, C = x.shape
    return x.reshape(N, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))

def vgg_forward(x: np.ndarray, config: list, kernels: list,
                biases: list, classifier: dict) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C_classes).
    """
    # 1. Feature Extraction
    features = np.asarray(x, dtype=np.float64)
    conv_idx = 0
    for layer in config:
        if layer == "M":
            features = _max_pool2d(features)
        else:
            kernel = np.asarray(kernels[conv_idx], dtype=np.float64)
            bias = np.asarray(biases[conv_idx], dtype=np.float64)
            features = _conv2d_relu(features, kernel, bias)
            conv_idx += 1

    # 2. Flatten (B, H, W, C) -> (B, H * W * C)
    batch_size = features.shape[0]
    flat = features.reshape(batch_size, -1)

    # 3. 3-Layer Classifier Head (FC1 + ReLU -> FC2 + ReLU -> FC3 Raw Logits)
    W1 = np.asarray(classifier["W1"], dtype=np.float64)
    b1 = np.asarray(classifier["b1"], dtype=np.float64)
    W2 = np.asarray(classifier["W2"], dtype=np.float64)
    b2 = np.asarray(classifier["b2"], dtype=np.float64)
    W3 = np.asarray(classifier["W3"], dtype=np.float64)
    b3 = np.asarray(classifier["b3"], dtype=np.float64)

    h1 = np.maximum(0.0, flat @ W1 + b1)
    h2 = np.maximum(0.0, h1 @ W2 + b2)
    logits = h2 @ W3 + b3

    return logits.astype(np.float64, copy=False)
