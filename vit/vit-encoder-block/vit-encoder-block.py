import numpy as np


def vit_encoder_block(
    x: np.ndarray,
    num_heads: int,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    Wo: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
) -> np.ndarray:
    """
    Returns the float64 output of one pre-normalized ViT encoder block.
    """
    x = np.asarray(x, dtype=np.float64)
    B, N, D = x.shape
    d_h = D // num_heads

    def layer_norm(t: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mean = np.mean(t, axis=-1, keepdims=True)
        var = np.var(t, axis=-1, keepdims=True)  # Population variance (ddof=0)
        return (t - mean) / np.sqrt(var + eps)

    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    def gelu(t: np.ndarray) -> np.ndarray:
        return 0.5 * t * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (t + 0.044715 * (t ** 3))))

    # 1. Pre-LayerNorm + Multi-Head Self-Attention + Residual
    x_norm = layer_norm(x)

    Q = (x_norm @ Wq).reshape(B, N, num_heads, d_h).transpose(0, 2, 1, 3)
    K = (x_norm @ Wk).reshape(B, N, num_heads, d_h).transpose(0, 2, 1, 3)
    V = (x_norm @ Wv).reshape(B, N, num_heads, d_h).transpose(0, 2, 1, 3)

    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_h)
    attn_weights = softmax(scores)

    # Concatenate heads back to (B, N, D) and project with Wo
    attn_out = (attn_weights @ V).transpose(0, 2, 1, 3).reshape(B, N, D)
    y = x + (attn_out @ Wo)

    # 2. Pre-LayerNorm + Two-Layer GELU MLP + Residual
    y_norm = layer_norm(y)
    mlp_out = gelu(y_norm @ W1) @ W2
    z = y + mlp_out

    return z.astype(np.float64)
