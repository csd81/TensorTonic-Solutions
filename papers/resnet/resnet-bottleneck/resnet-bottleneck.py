import numpy as np


def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns the bottleneck residual-block output as a nested list.
    """
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    W3 = np.asarray(W3, dtype=np.float64)

    # Bottleneck path: compress -> process -> expand
    h1 = np.maximum(0.0, x @ W1)
    h2 = np.maximum(0.0, h1 @ W2)
    out = h2 @ W3

    # Shortcut connection
    shortcut = x if Ws is None else x @ np.asarray(Ws, dtype=np.float64)

    # Combine residual path with shortcut and apply final ReLU
    y = np.maximum(0.0, out + shortcut)

    return np.round(y, 4).tolist()
