import numpy as np

def rnn_cell(X: list, h0: list, Wxh: list, Whh: list, bh: list, mode: str = "forward", dh_all=None) -> dict:
    """
    Returns hidden states and optional sequence gradients.
    """
    # Convert input lists to float64 numpy arrays
    X_arr = np.array(X, dtype=np.float64)
    h0_arr = np.array(h0, dtype=np.float64)
    Wxh_arr = np.array(Wxh, dtype=np.float64)
    Whh_arr = np.array(Whh, dtype=np.float64)
    bh_arr = np.array(bh, dtype=np.float64)
    
    T, D = X_arr.shape
    H = h0_arr.shape[0]
    
    # -------------------------------------------------------------------------
    # 1. Forward Pass
    # -------------------------------------------------------------------------
    # Store intermediate pre-activation values (z) and hidden states (h)
    # h will have shape (T + 1, H) to simplify indexing (h[-1] is h0)
    h_states = np.zeros((T + 1, H), dtype=np.float64)
    h_states[-1] = h0_arr
    z_states = np.zeros((T, H), dtype=np.float64)
    
    for t in range(T):
        x_t = X_arr[t]
        h_prev = h_states[t - 1]
        
        # z_t = Wxh * x_t + Whh * h_{t-1} + bh
        z_t = np.dot(Wxh_arr, x_t) + np.dot(Whh_arr, h_prev) + bh_arr
        z_states[t] = z_t
        h_states[t] = np.tanh(z_t)
        
    # Extract only the computed hidden states (excluding h0 placeholder at index -1)
    hidden_states_computed = h_states[:T]
    
    if mode == "forward":
        return {"hidden_states": np.round(hidden_states_computed, 4).tolist()}
        
    # -------------------------------------------------------------------------
    # 2. Backward Pass (Backpropagation Through Time)
    # -------------------------------------------------------------------------
    dh_all_arr = np.array(dh_all, dtype=np.float64)
    
    # Initialize gradient containers
    dWxh = np.zeros_like(Wxh_arr)
    dWhh = np.zeros_like(Whh_arr)
    dbh = np.zeros_like(bh_arr)
    dX = np.zeros_like(X_arr)
    
    # dh_next represents the incoming gradient from the subsequent time step (t+1)
    dh_next = np.zeros(H, dtype=np.float64)
    
    # Loop backwards through time from T-1 down to 0
    for t in reversed(range(T)):
        # Combined upstream gradient = direct upstream gradient + recurrent gradient from t+1
        dh_t = dh_all_arr[t] + dh_next
        
        # Derivative of tanh: dL/dz = dh_t * (1 - tanh^2(z_t)) = dh_t * (1 - h_t^2)
        h_t = h_states[t]
        dz_t = dh_t * (1.0 - h_t ** 2)
        
        # Local gradients for shared parameters at time step t
        x_t = X_arr[t]
        h_prev = h_states[t - 1]
        
        dWxh += np.outer(dz_t, x_t)
        dWhh += np.outer(dz_t, h_prev)
        dbh += dz_t
        
        # Gradients for input X at time step t
        dX[t] = np.dot(Wxh_arr.T, dz_t)
        
        # Update recurrent gradient to feed into the previous time step (t-1)
        dh_next = np.dot(Whh_arr.T, dz_t)
        
    return {
        "hidden_states": np.round(hidden_states_computed, 4).tolist(),
        "dWxh": np.round(dWxh, 4).tolist(),
        "dWhh": np.round(dWhh, 4).tolist(),
        "dbh": np.round(dbh, 4).tolist(),
        "dX": np.round(dX, 4).tolist()
    }
