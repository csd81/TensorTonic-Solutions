import numpy as np

def conv2d(x: list, W: list, b: list) -> np.ndarray:
    """
    Returns the convolved batch as a floating-point NumPy array.
    """
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    N, C_in, H, W_in = x.shape
    C_out, _, K_H, K_W = W.shape

    H_out = H - K_H + 1
    W_out = W_in - K_W + 1

    output = np.zeros((N, C_out, H_out, W_out), dtype=np.float64)

    for i in range(H_out):
        for j in range(W_out):
            # Extract spatial patch across all batches and input channels: (N, C_in, K_H, K_W)
            patch = x[:, :, i:i + K_H, j:j + K_W]
            
            # Multiply and sum over C_in, K_H, and K_W for all N and C_out simultaneously
            output[:, :, i, j] = np.tensordot(patch, W, axes=([1, 2, 3], [1, 2, 3])) + b

    return output
