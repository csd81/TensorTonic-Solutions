import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def lstm_forward(X: np.ndarray, W_f: np.ndarray, W_i: np.ndarray,
                 W_c: np.ndarray, W_o: np.ndarray, b_f: np.ndarray,
                 b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray,
                 W_y: np.ndarray, b_y: np.ndarray) -> dict:
    """
    Returns outputs, final_hidden_state, and final_cell_state.
    """
    X = np.asarray(X, dtype=np.float64)
    W_f = np.asarray(W_f, dtype=np.float64)
    W_i = np.asarray(W_i, dtype=np.float64)
    W_c = np.asarray(W_c, dtype=np.float64)
    W_o = np.asarray(W_o, dtype=np.float64)
    b_f = np.asarray(b_f, dtype=np.float64)
    b_i = np.asarray(b_i, dtype=np.float64)
    b_c = np.asarray(b_c, dtype=np.float64)
    b_o = np.asarray(b_o, dtype=np.float64)
    W_y = np.asarray(W_y, dtype=np.float64)
    b_y = np.asarray(b_y, dtype=np.float64)

    N, T, _ = X.shape
    H = b_f.shape[0]
    O = b_y.shape[0]

    h_t = np.zeros((N, H), dtype=np.float64)
    c_t = np.zeros((N, H), dtype=np.float64)
    outputs = np.zeros((N, T, O), dtype=np.float64)

    for t in range(T):
        x_t = X[:, t, :]
        concat = np.concatenate([h_t, x_t], axis=1)  # Shape: (N, H + D)

        f_t = _sigmoid(concat @ W_f.T + b_f)
        i_t = _sigmoid(concat @ W_i.T + b_i)
        c_tilde = np.tanh(concat @ W_c.T + b_c)
        o_t = _sigmoid(concat @ W_o.T + b_o)

        c_t = f_t * c_t + i_t * c_tilde
        h_t = o_t * np.tanh(c_t)

        outputs[:, t, :] = h_t @ W_y.T + b_y

    return {
        "outputs": outputs,
        "final_hidden_state": h_t,
        "final_cell_state": c_t,
    }
