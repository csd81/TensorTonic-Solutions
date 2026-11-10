import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lstm_cell(
    X: list, h0: list, c0: list, 
    Wf: list, Wi: list, Wc: list, Wo: list, 
    bf: list, bi: list, bc: list, bo: list, 
    mode: str = 'forward', 
    dh_all: list | None = None, 
    dc_last: list | None = None
) -> dict:
    # Convert all inputs to float64 numpy arrays
    X = np.array(X, dtype=np.float64)
    h0 = np.array(h0, dtype=np.float64)
    c0 = np.array(c0, dtype=np.float64)
    Wf = np.array(Wf, dtype=np.float64)
    Wi = np.array(Wi, dtype=np.float64)
    Wc = np.array(Wc, dtype=np.float64)
    Wo = np.array(Wo, dtype=np.float64)
    bf = np.array(bf, dtype=np.float64)
    bi = np.array(bi, dtype=np.float64)
    bc = np.array(bc, dtype=np.float64)
    bo = np.array(bo, dtype=np.float64)
    
    T, D = X.shape
    H = h0.shape[0]
    
    h_states = []
    c_states = []
    
    # Cache intermediate variables for the backward pass
    f_list = []
    i_list = []
    c_tilde_list = []
    o_list = []
    tanh_c_list = []
    concat_list = []
    
    h_prev = h0
    c_prev = c0
    
    # --- Forward Pass ---
    for t in range(T):
        xt = X[t]
        concat = np.concatenate([h_prev, xt])
        concat_list.append(concat)
        
        f = sigmoid(Wf @ concat + bf)
        i = sigmoid(Wi @ concat + bi)
        c_tilde = np.tanh(Wc @ concat + bc)
        o = sigmoid(Wo @ concat + bo)
        
        c_curr = f * c_prev + i * c_tilde
        tanh_c = np.tanh(c_curr)
        h_curr = o * tanh_c
        
        f_list.append(f)
        i_list.append(i)
        c_tilde_list.append(c_tilde)
        o_list.append(o)
        tanh_c_list.append(tanh_c)
        
        h_states.append(h_curr)
        c_states.append(c_curr)
        
        h_prev = h_curr
        c_prev = c_curr

    res = {
        'hidden_states': np.round(np.array(h_states), 4).tolist(),
        'cell_states': np.round(np.array(c_states), 4).tolist()
    }
    
    # --- Backward Pass ---
    if mode == 'backward':
        dh_all = np.array(dh_all, dtype=np.float64)
        dc_next = np.array(dc_last, dtype=np.float64) if dc_last is not None else np.zeros(H, dtype=np.float64)
            
        dWf = np.zeros_like(Wf)
        dWi = np.zeros_like(Wi)
        dWc = np.zeros_like(Wc)
        dWo = np.zeros_like(Wo)
        
        dbf = np.zeros_like(bf)
        dbi = np.zeros_like(bi)
        dbc = np.zeros_like(bc)
        dbo = np.zeros_like(bo)
        
        dX = np.zeros_like(X)
        dh_next = np.zeros(H, dtype=np.float64)
        
        for t in reversed(range(T)):
            c_prev = c_states[t-1] if t > 0 else c0
            
            dh = dh_all[t] + dh_next
            
            o = o_list[t]
            tanh_c = tanh_c_list[t]
            f = f_list[t]
            i = i_list[t]
            c_tilde = c_tilde_list[t]
            concat = concat_list[t]
            
            # Gradient flowing through gates
            do = dh * tanh_c
            dc = dh * o * (1 - tanh_c**2) + dc_next
            
            df = dc * c_prev
            di = dc * c_tilde
            dc_tilde = dc * i
            
            # Activation functions backprop
            d_o_act = do * o * (1 - o)
            d_f_act = df * f * (1 - f)
            d_i_act = di * i * (1 - i)
            d_c_tilde_act = dc_tilde * (1 - c_tilde**2)
            
            # Accumulate gradients for weights & biases
            dWf += np.outer(d_f_act, concat)
            dWi += np.outer(d_i_act, concat)
            dWc += np.outer(d_c_tilde_act, concat)
            dWo += np.outer(d_o_act, concat)
            
            dbf += d_f_act
            dbi += d_i_act
            dbc += d_c_tilde_act
            dbo += d_o_act
            
            # Backprop through concatenation map
            dconcat = (Wf.T @ d_f_act + Wi.T @ d_i_act + Wc.T @ d_c_tilde_act + Wo.T @ d_o_act)
            
            dh_next = dconcat[:H]
            dX[t] = dconcat[H:]
            dc_next = dc * f
            
        res.update({
            'dWf': np.round(dWf, 4).tolist(),
            'dWi': np.round(dWi, 4).tolist(),
            'dWc': np.round(dWc, 4).tolist(),
            'dWo': np.round(dWo, 4).tolist(),
            'dbf': np.round(dbf, 4).tolist(),
            'dbi': np.round(dbi, 4).tolist(),
            'dbc': np.round(dbc, 4).tolist(),
            'dbo': np.round(dbo, 4).tolist(),
            'dX': np.round(dX, 4).tolist()
        })
        
    return res
