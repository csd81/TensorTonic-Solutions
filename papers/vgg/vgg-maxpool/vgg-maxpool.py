import numpy as np


def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Returns the float64 pooled tensor in NHWC layout.
    """
    arr = np.asarray(x, dtype=np.float64)
    b, h, w, c = arr.shape

    # Split H into (H // 2, 2) and W into (W // 2, 2), then take max over window axes (2, 4)
    pooled = arr.reshape(b, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))

    return pooled.astype(np.float64)
