import numpy as np

def unet_bottleneck(x: np.ndarray, kernel1: np.ndarray, bias1: np.ndarray,
                    kernel2: np.ndarray, bias2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 bottleneck features in NHWC layout.
    """
    def apply_conv_same(tensor: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
        B, H, W, C_in = tensor.shape
        kh, kw, _, C_out = kernel.shape
        pad_h = kh // 2
        pad_w = kw // 2

        padded = np.pad(
            tensor,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode='constant',
            constant_values=0.0
        )

        out = np.zeros((B, H, W, C_out), dtype=np.float64)
        for i in range(H):
            for j in range(W):
                window = padded[:, i:i+kh, j:j+kw, :, np.newaxis]
                out[:, i, j, :] = np.sum(window * kernel[np.newaxis, :, :, :, :], axis=(1, 2, 3))

        out += bias
        return out

    # Ensure float64
    x = x.astype(np.float64, copy=False)
    kernel1 = kernel1.astype(np.float64, copy=False)
    bias1 = bias1.astype(np.float64, copy=False)
    kernel2 = kernel2.astype(np.float64, copy=False)
    bias2 = bias2.astype(np.float64, copy=False)

    # 1. First convolution + bias + ReLU
    c1 = apply_conv_same(x, kernel1, bias1)
    r1 = np.maximum(0.0, c1)

    # 2. Second convolution + bias + ReLU
    c2 = apply_conv_same(r1, kernel2, bias2)
    Y = np.maximum(0.0, c2).astype(np.float64, copy=False)

    return Y
