import numpy as np

def scaled_dot_product_attention(Q: list, K: list, V: list, mask: list | None = None, mode: str = 'forward', d_output: list | None = None) -> dict:
    """
    Returns forward results and optional backward gradients as a dictionary of lists.
    """
    # Convert inputs to float64 numpy arrays
    Q_arr = np.array(Q, dtype=np.float64)
    K_arr = np.array(K, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    d = Q_arr.shape[1]
    
    # 1. Compute scaled attention scores: S = (Q @ K.T) / sqrt(d)
    S = np.matmul(Q_arr, K_arr.T) / np.sqrt(d)
    
    # 2. Apply attention mask if provided
    if mask is not None:
        mask_arr = np.array(mask, dtype=bool)
        S = np.where(mask_arr, S, -1e9)
        
    # 3. Compute row-wise Softmax securely (subtract max for numerical stability)
    S_max = np.max(S, axis=-1, keepdims=True)
    exp_S = np.exp(S - S_max)
    A = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
    
    # 4. Compute output: O = A @ V
    O = np.matmul(A, V_arr)
    
    # Initialize response dictionary with rounded results
    result = {
        'output': np.round(O, 4).tolist(),
        'attention_weights': np.round(A, 4).tolist()
    }
    
    # 5. Backward Pass
    if mode == 'backward' and d_output is not None:
        dO = np.array(d_output, dtype=np.float64)
        
        # Gradients with respect to V and A from O = A @ V
        dV = np.matmul(A.T, dO)
        dA = np.matmul(dO, V_arr.T)
        
        # Gradient with respect to S from Softmax row-wise properties
        dS = A * (dA - np.sum(dA * A, axis=-1, keepdims=True))
        
        # Gradients with respect to Q and K from S = (Q @ K.T) / sqrt(d)
        dQ = np.matmul(dS, K_arr) / np.sqrt(d)
        dK = np.matmul(dS.T, Q_arr) / np.sqrt(d)
        
        # Add gradients rounded to 4 decimal places
        result['dQ'] = np.round(dQ, 4).tolist()
        result['dK'] = np.round(dK, 4).tolist()
        result['dV'] = np.round(dV, 4).tolist()
        
    return result
