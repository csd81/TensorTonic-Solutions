import numpy as np

def gru_cell(X: list, h0: list, Wz: list, Wr: list, Wh: list, bz: list, br: list, bh: list, mode: str = "forward", dh_all=None) -> dict:
    # Convert inputs to numpy float64 arrays
    X = np.array(X, dtype=np.float64)
    h0 = np.array(h0, dtype=np.float64)
    Wz = np.array(Wz, dtype=np.float64)
    Wr = np.array(Wr, dtype=np.float64)
    Wh = np.array(Wh, dtype=np.float64)
    bz = np.array(bz, dtype=np.float64)
    br = np.array(br, dtype=np.float64)
    bh = np.array(bh, dtype=np.float64)
    
    T, D = X.shape
    H = h0.shape[0]
    
    # Helper activation functions
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    # Storage for intermediate states for forward and backward pass
    h_states = []
    z_all, r_all, h_tilde_all, c_all, c_h_all = [], [], [], [], []
    
    h_prev = h0
    
    # --- Forward Pass ---
    for t in range(T):
        x_t = X[t]
        c_t = np.concatenate([h_prev, x_t])
        
        z_t = sigmoid(np.dot(Wz, c_t) + bz)
        r_t = sigmoid(np.dot(Wr, c_t) + br)
        
        c_h_t = np.concatenate([r_t * h_prev, x_t])
        h_tilde_t = np.tanh(np.dot(Wh, c_h_t) + bh)
        
        h_t = (1.0 - z_t) * h_prev + z_t * h_tilde_t
        
        # Save cache for BPTT
        z_all.append(z_t)
        r_all.append(r_t)
        h_tilde_all.append(h_tilde_t)
        c_all.append(c_t)
        c_h_all.append(c_h_t)
        h_states.append(h_t)
        
        h_prev = h_t
        
    if mode == "forward":
        return {"hidden_states": np.round(h_states, 4).tolist()}
        
    # --- Backward Pass (BPTT) ---
    dh_all = np.array(dh_all, dtype=np.float64)
    
    # Initialize gradients with zeros
    dWz = np.zeros_like(Wz)
    dWr = np.zeros_like(Wr)
    dWh = np.zeros_like(Wh)
    dbz = np.zeros_like(bz)
    dbr = np.zeros_like(br)
    dbh = np.zeros_like(bh)
    dX = np.zeros_like(X)
    
    dh_next = np.zeros(H)
    
    for t in reversed(range(T)):
        h_t = h_states[t]
        h_prev = h_states[t-1] if t > 0 else h0
        x_t = X[t]
        
        z_t = z_all[t]
        r_t = r_all[t]
        h_tilde_t = h_tilde_all[t]
        c_t = c_all[t]
        c_h_t = c_h_all[t]
        
        # Incoming gradient to h_t
        dh_curr = dh_all[t] + dh_next
        
        # Gradients of intermediate variables
        dh_tilde = dh_curr * z_t
        dz = dh_curr * (h_tilde_t - h_prev)
        dh_prev_from_interp = dh_curr * (1.0 - z_t)
        
        # Backprop through tanh to get gate input activations
        d_raw_h = dh_tilde * (1.0 - h_tilde_t ** 2)
        dWh += np.outer(d_raw_h, c_h_t)
        dbh += d_raw_h
        dc_h_t = np.dot(Wh.T, d_raw_h)
        
        dr_h_prev = dc_h_t[:H]
        dx_t_from_h = dc_h_t[H:]
        
        dr = dr_h_prev * h_prev
        dh_prev_from_h_tilde = dr_h_prev * r_t
        
        # Backprop through sigmoid gates
        d_raw_z = dz * z_t * (1.0 - z_t)
        dWz += np.outer(d_raw_z, c_t)
        dbz += d_raw_z
        dc_t_from_z = np.dot(Wz.T, d_raw_z)
        
        d_raw_r = dr * r_t * (1.0 - r_t)
        dWr += np.outer(d_raw_r, c_t)
        dbr += d_raw_r
        dc_t_from_r = np.dot(Wr.T, d_raw_r)
        
        # Sum up contributions for c_t = [h_prev, x_t]
        dc_t = dc_t_from_z + dc_t_from_r
        dh_prev_from_gates = dc_t[:H]
        dx_t_from_gates = dc_t[H:]
        
        # Collect final gradients for dX and dh_next
        dX[t] = dx_t_from_h + dx_t_from_gates
        dh_next = dh_prev_from_interp + dh_prev_from_h_tilde + dh_prev_from_gates

    return {
        "hidden_states": np.round(h_states, 4).tolist(),
        "dWz": np.round(dWz, 4).tolist(),
        "dWr": np.round(dWr, 4).tolist(),
        "dWh": np.round(dWh, 4).tolist(),
        "dbz": np.round(dbz, 4).tolist(),
        "dbr": np.round(dbr, 4).tolist(),
        "dbh": np.round(dbh, 4).tolist(),
        "dX": np.round(dX, 4).tolist()
    }
