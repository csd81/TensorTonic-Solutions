import numpy as np

def multi_head_attention(Q: list, K: list, V: list, W_q: list, W_k: list, W_v: list, W_o: list, num_heads: int, mask: list | None = None) -> dict:
    """
    Computes Multi-Head Attention with given weights and dimensions.
    Returns a dictionary with 'output' and 'attention_weights' rounded to 4 decimal places.
    """
    # 1. Convert all inputs to float64 NumPy arrays
    Q_arr = np.array(Q, dtype=np.float64)
    K_arr = np.array(K, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    
    W_q_arr = np.array(W_q, dtype=np.float64)
    W_k_arr = np.array(W_k, dtype=np.float64)
    W_v_arr = np.array(W_v, dtype=np.float64)
    W_o_arr = np.array(W_o, dtype=np.float64)
    
    T, D = Q_arr.shape
    d_h = D // num_heads
    sqrt_dh = np.sqrt(d_h)
    
    # 2. Linear Projections using matrix transpose: X @ W^T
    Q_proj = Q_arr @ W_q_arr.T
    K_proj = K_arr @ W_k_arr.T
    V_proj = V_arr @ W_v_arr.T
    
    # 3. Split projected features into heads
    # Reshape from (T, D) -> (T, num_heads, d_h) -> Transpose to (num_heads, T, d_h)
    Q_heads = Q_proj.reshape(T, num_heads, d_h).transpose(1, 0, 2)
    K_heads = K_proj.reshape(T, num_heads, d_h).transpose(1, 0, 2)
    V_heads = V_proj.reshape(T, num_heads, d_h).transpose(1, 0, 2)
    
    # 4. Compute Scaled Dot-Product Attention for each head
    # Scores shape: (num_heads, T, T)
    scores = (Q_heads @ K_heads.transpose(0, 2, 1)) / sqrt_dh
    
    # Apply attention mask if provided
    if mask is not None:
        mask_arr = np.array(mask, dtype=bool)
        # Broadcast mask over head dimension
        scores = np.where(mask_arr, scores, -1e9)
        
    # Softmax over key positions (last dimension)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    A = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 5. Compute Head Outputs
    # O_heads shape: (num_heads, T, d_h)
    O_heads = A @ V_heads
    
    # 6. Concatenate heads back together
    # Transpose back to (T, num_heads, d_h) -> Reshape to (T, D)
    O_concat = O_heads.transpose(1, 0, 2).reshape(T, D)
    
    # 7. Apply Output Projection: O_concat @ W_o^T
    output = O_concat @ W_o_arr.T
    
    # 8. Round to 4 decimal places and convert back to native nested Python lists
    return {
        "output": np.round(output, 4).tolist(),
        "attention_weights": np.round(A, 4).tolist()
    }
