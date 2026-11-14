import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray, 
               W_i: np.ndarray, b_i: np.ndarray, 
               W_c: np.ndarray, b_c: np.ndarray) -> dict:
    """
    Returns input_gate and candidate_state as float64 arrays.
    """
    # Convert inputs to float64 arrays in case Python lists are passed
    h_prev = np.asarray(h_prev, dtype=np.float64)
    x_t = np.asarray(x_t, dtype=np.float64)
    
    # axis=-1 works for both 1D (H,) + (D,) and 2D (N, H) + (N, D)
    joined = np.concatenate([h_prev, x_t], axis=-1)
    
    i_t = sigmoid(joined @ W_i.T + b_i).astype(np.float64)
    c_tilde_t = np.tanh(joined @ W_c.T + b_c).astype(np.float64)

    return {
        "input_gate": i_t,
        "candidate_state": c_tilde_t
    }
