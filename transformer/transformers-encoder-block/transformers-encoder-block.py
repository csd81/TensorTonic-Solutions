import numpy as np

def softmax(x, axis=-1):
    """Provided: numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns x normalized over the last axis, then scaled by gamma and shifted by beta.
    """
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True) # Population variance (ddof=0)
    
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns multi-head attention of Q, K, V, projecting them with W_q, W_k, W_v and the merged heads with W_o.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W_q = np.asarray(W_q, dtype=np.float64)
    W_k = np.asarray(W_k, dtype=np.float64)
    W_v = np.asarray(W_v, dtype=np.float64)
    W_o = np.asarray(W_o, dtype=np.float64)
    
    batch_size, seq_len_q, d_model = Q.shape
    _, seq_len_k, _ = K.shape
    d_k = d_model // num_heads
    
    # Linear projections: (batch, seq_len, d_model)
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v
    
    # Reshape and transpose for multi-head: (batch, num_heads, seq_len, d_k)
    Q_heads = Q_proj.reshape(batch_size, seq_len_q, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch_size, seq_len_k, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch_size, seq_len_k, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # Scaled dot-product attention
    scores = (Q_heads @ K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attn_weights = softmax(scores, axis=-1)
    context = attn_weights @ V_heads
    
    # Concatenate heads: (batch, seq_len_q, d_model)
    context_concat = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, d_model)
    
    # Final output projection
    return context_concat @ W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward network output with a ReLU hidden layer.
    """
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)
    
    hidden = np.maximum(0.0, x @ W1 + b1)
    return hidden @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Sub-layer 1: Multi-Head Self-Attention + Residual + LayerNorm
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    z = layer_norm(x + attn_out, gamma1, beta1)
    
    # Sub-layer 2: Feed-Forward Network + Residual + LayerNorm
    ffn_out = feed_forward(z, W1, b1, W2, b2)
    y = layer_norm(z + ffn_out, gamma2, beta2)
    
    return y
