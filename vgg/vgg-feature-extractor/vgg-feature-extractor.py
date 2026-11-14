import numpy as np

def _conv2d_relu(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    N, H, W, C_in = x.shape
    K_h, K_w, _, C_out = kernel.shape
    
    # Calculate asymmetric/symmetric "same" padding
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
            # Contract over (K_h, K_w, C_in) -> (N, C_out)
            out[:, h, w, :] = np.tensordot(patch, kernel, axes=([1, 2, 3], [0, 1, 2])) + bias
            
    return np.maximum(0.0, out)

def _max_pool2d(x: np.ndarray) -> np.ndarray:
    N, H, W, C = x.shape
    # Reshape into non-overlapping 2x2 spatial blocks and take the max
    return x.reshape(N, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))

def vgg_features(x: np.ndarray, config: list,
                 kernels: list, biases: list) -> np.ndarray:
    """
    Returns the float64 VGG features in NHWC layout.
    """
    out = np.asarray(x, dtype=np.float64)
    conv_idx = 0
    
    for layer in config:
        if layer == "M":
            out = _max_pool2d(out)
        else:
            kernel = np.asarray(kernels[conv_idx], dtype=np.float64)
            bias = np.asarray(biases[conv_idx], dtype=np.float64)
            out = _conv2d_relu(out, kernel, bias)
            conv_idx += 1
            
    return out.astype(np.float64, copy=False)
