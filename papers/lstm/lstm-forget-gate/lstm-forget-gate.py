import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    """
    Returns the float64 forget-gate values.
    """
    # Convert inputs to float64 arrays to ensure precision and handle list inputs
    h_prev = np.asarray(h_prev, dtype=np.float64)
    x_t = np.asarray(x_t, dtype=np.float64)
    W_f = np.asarray(W_f, dtype=np.float64)
    b_f = np.asarray(b_f, dtype=np.float64)
    
    # Concatenate h_prev and x_t along the last axis
    concat = np.concatenate([h_prev, x_t], axis=-1)
    
    # Compute the linear transformation and apply sigmoid
    logits = concat @ W_f.T + b_f
    return sigmoid(logits).astype(np.float64)
