import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def output_gate(h_prev: np.ndarray, x_t: np.ndarray, C_t: np.ndarray,
                W_o: np.ndarray, b_o: np.ndarray) -> dict:
    """
    Returns output_gate and hidden_state as float64 arrays.
    """
    h_prev = np.asarray(h_prev, dtype=np.float64)
    x_t = np.asarray(x_t, dtype=np.float64)
    C_t = np.asarray(C_t, dtype=np.float64)
    
    joined = np.concatenate([h_prev, x_t], axis=-1)
    
    o_t = sigmoid(joined @ W_o.T + b_o).astype(np.float64)
    h_t = (o_t * np.tanh(C_t)).astype(np.float64)
    
    return {
        "output_gate": o_t,
        "hidden_state": h_t
    }
