import numpy as np

def rnn_step_forward(x_t: list, h_prev: list, Wx: list, Wh: list, b: list) -> np.ndarray:
    """
    Returns a NumPy array with shape (H,).
    """
    x = np.asarray(x_t, dtype=np.float64)
    h = np.asarray(h_prev, dtype=np.float64)
    Wx_arr = np.asarray(Wx, dtype=np.float64)
    Wh_arr = np.asarray(Wh, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)

    a_t = x @ Wx_arr + h @ Wh_arr + b_arr
    h_t = np.tanh(a_t)

    return h_t
