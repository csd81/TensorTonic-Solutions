import numpy as np

def rnn_step_backward(dh: list, cache: list) -> dict:
    """
    Returns a dictionary with dx_t, dh_prev, dW, dU, and db.
    """
    # Unpack cache and convert inputs to NumPy float arrays
    x_t, h_prev, h_t, W, U, b = cache
    dh = np.asarray(dh, dtype=float)
    x_t = np.asarray(x_t, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    h_t = np.asarray(h_t, dtype=float)
    W = np.asarray(W, dtype=float)
    U = np.asarray(U, dtype=float)

    # 1. Backpropagate through the tanh activation
    dz = dh * (1.0 - h_t ** 2)

    # 2. Gradients with respect to inputs (x_t and h_prev)
    dx_t = W.T @ dz
    dh_prev = U.T @ dz

    # 3. Gradients with respect to parameters (W, U, and b)
    dW = np.outer(dz, x_t)
    dU = np.outer(dz, h_prev)
    db = dz.copy()

    return {
        "dx_t": dx_t,
        "dh_prev": dh_prev,
        "dW": dW,
        "dU": dU,
        "db": db,
    }
