import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))

def reset_gate(h_prev: np.ndarray, x_t: np.ndarray, W_r: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    """
    Returns the float64 reset-gate values.
    """
    # Ensure inputs are at least 2D for unified handling, or handle unbatched/batched cleanly
    h_prev_is_1d = h_prev.ndim == 1
    if h_prev_is_1d:
        h_prev = h_prev[np.newaxis, :]
        x_t = x_t[np.newaxis, :]
    
    # Concatenate h_prev and x_t along the last axis (axis=-1)
    # Shape becomes (N, H + D)
    concat_input = np.concatenate([h_prev, x_t], axis=-1)
    
    # Linear transformation: W_r @ [h_prev, x_t]^T + b_r
    # W_r has shape (H, H + D), so concat_input @ W_r.T gives (N, H)
    out = np.dot(concat_input, W_r.T) + b_r
    
    # Apply sigmoid
    r_t = sigmoid(out)
    
    # If input was unbatched 1D, return 1D float64 array
    if h_prev_is_1d:
        return r_t.squeeze(axis=0).astype(np.float64, copy=False)
        
    return r_t.astype(np.float64, copy=False)
