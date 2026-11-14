import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> dict:
    """
    Returns hidden_state and cell_state as float64 arrays.
    """
    x_t = np.asarray(x_t, dtype=np.float64)
    h_prev = np.asarray(h_prev, dtype=np.float64)
    C_prev = np.asarray(C_prev, dtype=np.float64)
    
    # Concatenate [h_{t-1}, x_t] along the last axis (handles both 1D and 2D)
    joined = np.concatenate([h_prev, x_t], axis=-1)
    
    # 1. Compute gates and candidate memory
    f_t = sigmoid(joined @ W_f.T + b_f)
    i_t = sigmoid(joined @ W_i.T + b_i)
    c_tilde = np.tanh(joined @ W_c.T + b_c)
    o_t = sigmoid(joined @ W_o.T + b_o)
    
    # 2. Update cell state
    C_t = (f_t * C_prev) + (i_t * c_tilde)
    
    # 3. Compute new hidden state
    h_t = o_t * np.tanh(C_t)
    
    return {
        "hidden_state": h_t.astype(np.float64),
        "cell_state": C_t.astype(np.float64)
    }
