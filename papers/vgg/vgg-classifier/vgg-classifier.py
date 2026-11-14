import numpy as np


def vgg_classifier(
    features: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    W3: np.ndarray,
    b3: np.ndarray,
) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C_classes).
    """
    x = np.asarray(features, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)
    W3 = np.asarray(W3, dtype=np.float64)
    b3 = np.asarray(b3, dtype=np.float64)

    # Flatten (B, H, W, C) -> (B, H * W * C) in row-major order
    flat = x.reshape(x.shape[0], -1)

    # First two fully connected layers with ReLU
    h1 = np.maximum(0.0, flat @ W1 + b1)
    h2 = np.maximum(0.0, h1 @ W2 + b2)

    # Final linear projection to raw class logits
    logits = h2 @ W3 + b3

    return logits.astype(np.float64)
