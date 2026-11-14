import numpy as np

def vanilla_rnn(X: np.ndarray, h_0: np.ndarray, W_xh: np.ndarray,
                W_hh: np.ndarray, W_hy: np.ndarray, b_h: np.ndarray,
                b_y: np.ndarray) -> dict:
    """
    Returns outputs and final_hidden_state as float64 arrays.
    """
    X = np.asarray(X, dtype=np.float64)
    h_t = np.asarray(h_0, dtype=np.float64).copy()
    W_xh = np.asarray(W_xh, dtype=np.float64)
    W_hh = np.asarray(W_hh, dtype=np.float64)
    W_hy = np.asarray(W_hy, dtype=np.float64)
    b_h = np.asarray(b_h, dtype=np.float64)
    b_y = np.asarray(b_y, dtype=np.float64)

    N, T, _ = X.shape
    O = W_hy.shape[0]

    outputs = np.zeros((N, T, O), dtype=np.float64)

    for t in range(T):
        x_t = X[:, t, :]
        # h_t = tanh(x_t @ W_xh.T + h_{t-1} @ W_hh.T + b_h)
        h_t = np.tanh(x_t @ W_xh.T + h_t @ W_hh.T + b_h)
        # y_t = h_t @ W_hy.T + b_y
        outputs[:, t, :] = h_t @ W_hy.T + b_y

    return {
        "outputs": outputs,
        "final_hidden_state": h_t
    }
