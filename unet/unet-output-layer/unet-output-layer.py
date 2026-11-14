import numpy as np

def unet_output(features: np.ndarray, W_out: np.ndarray,
                b_out: np.ndarray) -> np.ndarray:
    """
    Returns float64 per-pixel class logits in NHWC layout.
    """
    features = features.astype(np.float64, copy=False)
    W_out = W_out.astype(np.float64, copy=False)
    b_out = b_out.astype(np.float64, copy=False)

    # Apply 1x1 convolution (per-pixel linear projection) and add class bias
    logits = np.matmul(features, W_out) + b_out
    return logits.astype(np.float64, copy=False)
