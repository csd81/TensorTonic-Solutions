import numpy as np

def unet_decoder_block(
    x: np.ndarray, 
    skip: np.ndarray,
    W_up: np.ndarray, 
    b_up: np.ndarray,
    kernel1: np.ndarray, 
    bias1: np.ndarray,
    kernel2: np.ndarray, 
    bias2: np.ndarray
) -> np.ndarray:
    """
    Returns the float64 decoder features in NHWC layout.
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
    skip = skip.astype(np.float64, copy=False)
    W_up = W_up.astype(np.float64, copy=False)
    b_up = b_up.astype(np.float64, copy=False)
    kernel1 = kernel1.astype(np.float64, copy=False)
    bias1 = bias1.astype(np.float64, copy=False)
    kernel2 = kernel2.astype(np.float64, copy=False)
    bias2 = bias2.astype(np.float64, copy=False)

    # 1. Upsample spatial dimensions by repeating twice along H and W
    upsampled = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)

    # 2. Apply channel projection and ReLU: U = ReLU(Upsample(X) @ W_up + b_up)
    U = np.maximum(0.0, np.matmul(upsampled, W_up) + b_up)

    # 3. Center-crop the skip tensor to match U's spatial dimensions
    _, target_H, target_W, _ = U.shape
    _, skip_H, skip_W, _ = skip.shape
    start_h = (skip_H - target_H) // 2
    start_w = (skip_W - target_W) // 2
    cropped_skip = skip[:, start_h:start_h + target_H, start_w:start_w + target_W, :]

    # 4. Concatenate skip channels before decoder channels along axis=-1
    fused = np.concatenate([cropped_skip, U], axis=-1)

    # 5. Apply two same-padded convolutions with ReLU
    c1 = apply_conv_same(fused, kernel1, bias1)
    r1 = np.maximum(0.0, c1)

    c2 = apply_conv_same(r1, kernel2, bias2)
    Y = np.maximum(0.0, c2).astype(np.float64, copy=False)

    return Y
