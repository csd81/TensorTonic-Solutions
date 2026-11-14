import numpy as np

def unet_forward(x: np.ndarray, weights: dict) -> np.ndarray:
    """
    Returns float64 per-pixel segmentation logits in NHWC layout.
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

    def conv_block(tensor: np.ndarray, k1: np.ndarray, b1: np.ndarray, k2: np.ndarray, b2: np.ndarray) -> np.ndarray:
        r1 = np.maximum(0.0, apply_conv_same(tensor, k1, b1))
        r2 = np.maximum(0.0, apply_conv_same(r1, k2, b2))
        return r2

    # Ensure float64 for input and all weights
    x = x.astype(np.float64, copy=False)
    w = {k: v.astype(np.float64, copy=False) for k, v in weights.items()}

    # 1. Encoder Block (save skip tensor, then 2x2 non-overlapping max pool)
    skip = conv_block(x, w["enc_kernel1"], w["enc_bias1"], w["enc_kernel2"], w["enc_bias2"])
    
    B, H, W, C = skip.shape
    pooled_h, pooled_w = H // 2, W // 2
    pooled = np.zeros((B, pooled_h, pooled_w, C), dtype=np.float64)
    for i in range(pooled_h):
        for j in range(pooled_w):
            pooled[:, i, j, :] = np.max(skip[:, i*2:(i+1)*2, j*2:(j+1)*2, :], axis=(1, 2))

    # 2. Bottleneck (two same-padded convolutions with ReLU at lowest resolution)
    bottleneck = conv_block(
        pooled, 
        w["bridge_kernel1"], w["bridge_bias1"], 
        w["bridge_kernel2"], w["bridge_bias2"]
    )

    # 3. Decoder Block
    # 3a. Nearest-neighbor 2x upsampling + channel projection + ReLU
    upsampled = np.repeat(np.repeat(bottleneck, 2, axis=1), 2, axis=2)
    U = np.maximum(0.0, np.matmul(upsampled, w["W_up"]) + w["b_up"])

    # 3b. Center-crop skip tensor & concatenate before decoder channels
    _, target_h, target_w, _ = U.shape
    _, skip_h, skip_w, _ = skip.shape
    start_h = (skip_h - target_h) // 2
    start_w = (skip_w - target_w) // 2
    cropped_skip = skip[:, start_h:start_h + target_h, start_w:start_w + target_w, :]
    fused = np.concatenate([cropped_skip, U], axis=-1)

    # 3c. Decoder two-convolution block with ReLU
    dec_out = conv_block(
        fused, 
        w["dec_kernel1"], w["dec_bias1"], 
        w["dec_kernel2"], w["dec_bias2"]
    )

    # 4. Output Layer (1x1 per-pixel linear projection to raw logits)
    logits = np.matmul(dec_out, w["W_out"]) + w["b_out"]
    return logits.astype(np.float64, copy=False)
