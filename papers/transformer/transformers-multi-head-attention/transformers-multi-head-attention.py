import numpy as np


def multi_head_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_o: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    batch_size, q_len, d_model = Q.shape
    _, k_len, _ = K.shape
    d_k = d_model // num_heads

    # 1. Project Q, K, and V
    Q_proj = Q @ W_q  # (batch, q_len, d_model)
    K_proj = K @ W_k  # (batch, k_len, d_model)
    V_proj = V @ W_v  # (batch, k_len, d_model)

    # 2. Split into num_heads and transpose for parallel head computation
    # New shape: (batch, num_heads, seq_len, d_k)
    Q_heads = Q_proj.reshape(batch_size, q_len, num_heads, d_k).transpose(
        0, 2, 1, 3
    )
    K_heads = K_proj.reshape(batch_size, k_len, num_heads, d_k).transpose(
        0, 2, 1, 3
    )
    V_heads = V_proj.reshape(batch_size, k_len, num_heads, d_k).transpose(
        0, 2, 1, 3
    )

    # 3. Scaled dot-product attention
    # Scores shape: (batch, num_heads, q_len, k_len)
    scores = (Q_heads @ K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

    # 4. Numerically stable softmax over key positions (last axis)
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Weighted sum of values -> shape: (batch, num_heads, q_len, d_k)
    head_outputs = attn_weights @ V_heads

    # 5. Concatenate heads back to (batch, q_len, d_model)
    concat = head_outputs.transpose(0, 2, 1, 3).reshape(
        batch_size, q_len, d_model
    )

    # 6. Apply final output projection W_o
    output = concat @ W_o

    return output.astype(np.float64)
