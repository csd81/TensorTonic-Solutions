import numpy as np

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid computation
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

def gru_cell_forward(x: list, h_prev: list, params: dict) -> np.ndarray:
    """
    Returns the updated hidden state as a NumPy array matching the shape of h_prev.
    """
    # Convert inputs to float arrays without mutating the originals
    x_arr = np.asarray(x, dtype=np.float64)
    h_arr = np.asarray(h_prev, dtype=np.float64)
    
    # Track if input is 1D so we can restore the exact shape at the end
    is_1d = (h_arr.ndim == 1)
    if is_1d:
        x_arr = x_arr[np.newaxis, :]
        h_arr = h_arr[np.newaxis, :]
        
    # Extract parameters as float arrays
    Wz = np.asarray(params["Wz"], dtype=np.float64)
    Uz = np.asarray(params["Uz"], dtype=np.float64)
    bz = np.asarray(params["bz"], dtype=np.float64)
    
    Wr = np.asarray(params["Wr"], dtype=np.float64)
    Ur = np.asarray(params["Ur"], dtype=np.float64)
    br = np.asarray(params["br"], dtype=np.float64)
    
    Wh = np.asarray(params["Wh"], dtype=np.float64)
    Uh = np.asarray(params["Uh"], dtype=np.float64)
    bh = np.asarray(params["bh"], dtype=np.float64)
    
    # 1. Update gate: z_t = sigmoid(x_t @ W_z + h_{t-1} @ U_z + b_z)
    z_t = _sigmoid(x_arr @ Wz + h_arr @ Uz + bz)
    
    # 2. Reset gate: r_t = sigmoid(x_t @ W_r + h_{t-1} @ U_r + b_r)
    r_t = _sigmoid(x_arr @ Wr + h_arr @ Ur + br)
    
    # 3. Candidate hidden state: h_tilde = tanh(x_t @ W_h + (r_t * h_{t-1}) @ U_h + b_h)
    h_tilde = np.tanh(x_arr @ Wh + (r_t * h_arr) @ Uh + bh)
    
    # 4. New hidden state: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
    h_next = (1.0 - z_t) * h_arr + z_t * h_tilde
    
    # Restore 1D shape if the input was a single sample
    if is_1d:
        return h_next[0]
        
    return h_next
