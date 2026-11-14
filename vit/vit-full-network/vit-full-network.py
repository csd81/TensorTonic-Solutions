import numpy as np


def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def _gelu(x: np.ndarray) -> np.ndarray:
    # Standard tanh approximation used in pure NumPy implementations
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x**3))))


def _softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def vit_forward(
    image: np.ndarray,
    patch_size: int,
    num_heads: int,
    W_patch: np.ndarray,
    patch_bias: np.ndarray,
    cls_token: np.ndarray,
    pos_embed: np.ndarray,
    encoder_weights: list,
    W_head: np.ndarray,
) -> np.ndarray:
    """
    Returns float64 Vision Transformer logits with shape (B, C).
    """
    B, H, W, C_in = image.shape
    grid_h = H // patch_size
    grid_w = W // patch_size
    num_patches = grid_h * grid_w

    # 1. Extract complete non-overlapping patches in row-major order
    cropped = image[:, : grid_h * patch_size, : grid_w * patch_size, :]
    patches = cropped.reshape(B, grid_h, patch_size, grid_w, patch_size, C_in)
    patches = patches.transpose(0, 1, 3, 2, 4, 5).reshape(
        B, num_patches, patch_size * patch_size * C_in
    )

    # 2. Linear patch projection
    x = patches @ W_patch + patch_bias  # (B, N, D)
    D = x.shape[-1]

    # 3. Prepend [CLS] token and add positional embeddings
    cls_tokens = np.broadcast_to(cls_token, (B, 1, D))
    x = np.concatenate([cls_tokens, x], axis=1)  # (B, N + 1, D)
    x = x + pos_embed

    # 4. Transformer Encoder blocks (Pre-LN architecture)
    head_dim = D // num_heads
    seq_len = num_patches + 1

    for block in encoder_weights:
        # Multi-Head Self-Attention (MSA)
        norm1 = _layer_norm(x)
        Q = (norm1 @ block["Wq"]).reshape(B, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = (norm1 @ block["Wk"]).reshape(B, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        V = (norm1 @ block["Wv"]).reshape(B, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        attn_weights = _softmax(scores)
        attn_out = (attn_weights @ V).transpose(0, 2, 1, 3).reshape(B, seq_len, D)
        x = x + (attn_out @ block["Wo"])

        # Feed-Forward Network (MLP)
        norm2 = _layer_norm(x)
        mlp_out = _gelu(norm2 @ block["W1"]) @ block["W2"]
        x = x + mlp_out

    # 5. Classification Head on final [CLS] representation
    cls_final = _layer_norm(x[:, 0, :])  # (B, D)
    logits = cls_final @ W_head  # (B, C)

    return logits.astype(np.float64)
