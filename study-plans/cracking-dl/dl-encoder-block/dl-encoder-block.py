import numpy as np

def encoder_block(
    x: list, 
    W_q: list, 
    W_k: list, 
    W_v: list, 
    W_o: list, 
    num_heads: int, 
    W1: list, 
    b1: list, 
    W2: list, 
    b2: list, 
    gamma1: list, 
    beta1: list, 
    gamma2: list, 
    beta2: list
) -> dict:
    """
    Implements a post-normalization Transformer encoder block using float64.
    Returns a dictionary containing rounded lists: attention_output, norm1, ffn_output, and output.
    """
    # Convert all inputs to NumPy float64 arrays
    x = np.array(x, dtype=np.float64)
    W_q = np.array(W_q, dtype=np.float64)
    W_k = np.array(W_k, dtype=np.float64)
    W_v = np.array(W_v, dtype=np.float64)
    W_o = np.array(W_o, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    b1 = np.array(b1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    b2 = np.array(b2, dtype=np.float64)
    gamma1 = np.array(gamma1, dtype=np.float64)
    beta1 = np.array(beta1, dtype=np.float64)
    gamma2 = np.array(gamma2, dtype=np.float64)
    beta2 = np.array(beta2, dtype=np.float64)
    
    T, D = x.shape
    d_k = D // num_heads
    
    # 1. Multi-head Attention Projections
    Q = x @ W_q.T  # (T, D)
    K = x @ W_k.T  # (T, D)
    V = x @ W_v.T  # (T, D)
    
    # Reshape and transpose to isolate heads: (num_heads, T, d_k)
    Q_heads = Q.reshape(T, num_heads, d_k).transpose(1, 0, 2)
    K_heads = K.reshape(T, num_heads, d_k).transpose(1, 0, 2)
    V_heads = V.reshape(T, num_heads, d_k).transpose(1, 0, 2)
    
    head_outputs = []
    for h in range(num_heads):
        Q_h = Q_heads[h]
        K_h = K_heads[h]
        V_h = V_heads[h]
        
        # Scaled dot-product attention
        scores = (Q_h @ K_h.T) / np.sqrt(d_k)
        
        # Row-wise stable Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        head_out = attn_weights @ V_h
        head_outputs.append(head_out)
        
    # Concatenate head outputs along the last dimension
    concat_out = np.concatenate(head_outputs, axis=-1)
    attention_output = concat_out @ W_o.T
    
    # Helper for Population Layer Normalization
    def layer_norm(z, gamma, beta):
        mean = np.mean(z, axis=-1, keepdims=True)
        var = np.var(z, axis=-1, keepdims=True)  # population variance (ddof=0)
        return ((z - mean) / np.sqrt(var + 1e-5)) * gamma + beta

    # 2. First Residual and Layer Norm (Post-LN)
    norm1 = layer_norm(x + attention_output, gamma1, beta1)
    
    # 3. Feed-Forward Network Block
    ffn_hidden = np.maximum(0, norm1 @ W1.T + b1)  # ReLU
    ffn_output = ffn_hidden @ W2.T + b2
    
    # 4. Second Residual and Layer Norm
    output = layer_norm(norm1 + ffn_output, gamma2, beta2)
    
    # Format precision and structures into Python lists
    return {
        "attention_output": np.round(attention_output, 4).tolist(),
        "norm1": np.round(norm1, 4).tolist(),
        "ffn_output": np.round(ffn_output, 4).tolist(),
        "output": np.round(output, 4).tolist()
    }
