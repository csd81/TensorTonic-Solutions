import numpy as np

def update_cell_state(C_prev: np.ndarray, f_t: np.ndarray,
                      i_t: np.ndarray, c_tilde: np.ndarray) -> np.ndarray:
    """
    Returns the float64 updated cell state.
    """
    C_prev = np.asarray(C_prev, dtype=np.float64)
    f_t = np.asarray(f_t, dtype=np.float64)
    i_t = np.asarray(i_t, dtype=np.float64)
    c_tilde = np.asarray(c_tilde, dtype=np.float64)
    
    C_t = (f_t * C_prev) + (i_t * c_tilde)
    return C_t.astype(np.float64)
