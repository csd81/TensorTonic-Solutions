import math
import torch
import torch.nn.functional as F

def rms_norm(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * gain

def apply_rope(x: torch.Tensor, cos_freq: torch.Tensor, sin_freq: torch.Tensor) -> torch.Tensor:
    # x: (B, H, S, D_h)
    # cos_freq, sin_freq: (S, D_h // 2) -> expand to (1, 1, S, D_h // 2)
    cos = cos_freq.unsqueeze(0).unsqueeze(0)
    sin = sin_freq.unsqueeze(0).unsqueeze(0)
    
    # Split half (Gemma / HuggingFace standard RoPE convention)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    
    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x1 * sin + x2 * cos
    return torch.cat([rot_x1, rot_x2], dim=-1)

def gemma3_attention(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    cos_freq: torch.Tensor,
    sin_freq: torch.Tensor,
    layer_idx: int,
    local_ratio: int,
    window_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Returns the float64 output of one Gemma 3 attention sublayer.
    """
    # Ensure float64 across inputs and weights
    x = x.to(torch.float64)
    w = {k: v.to(torch.float64) for k, v in weights.items()}
    cos_freq = cos_freq.to(torch.float64)
    sin_freq = sin_freq.to(torch.float64)

    B, S, D = x.shape
    head_dim = D // num_query_heads

    # 1. Pre-attention RMSNorm
    h = rms_norm(x, w["pre_attention_gain"], eps)

    # 2. Q, K, V projections (weights are stored as out_features x in_features)
    q = F.linear(h, w["query_projection"])
    k = F.linear(h, w["key_projection"])
    v = F.linear(h, w["value_projection"])

    # 3. Reshape and transpose to (B, num_heads, S, head_dim)
    q = q.view(B, S, num_query_heads, head_dim).transpose(1, 2)
    k = k.view(B, S, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(B, S, num_kv_heads, head_dim).transpose(1, 2)

    # 4. QK-Norm along head_dim
    q = rms_norm(q, w["query_gain"], eps)
    k = rms_norm(k, w["key_gain"], eps)

    # 5. Apply Rotary Position Embeddings (RoPE)
    q = apply_rope(q, cos_freq, sin_freq)
    k = apply_rope(k, cos_freq, sin_freq)

    # 6. Grouped-Query Attention (GQA): repeat KV heads to match query heads
    if num_query_heads != num_kv_heads:
        num_groups = num_query_heads // num_kv_heads
        k = k.repeat_interleave(num_groups, dim=1)
        v = v.repeat_interleave(num_groups, dim=1)

    # 7. Scaled Dot-Product Attention with Global vs. Local Causal Mask
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    pos = torch.arange(S, device=x.device)
    i = pos.unsqueeze(1)  # Query positions (S, 1)
    j = pos.unsqueeze(0)  # Key positions (1, S)

    is_global = ((layer_idx + 1) % (local_ratio + 1)) == 0
    if is_global:
        valid_mask = j <= i
    else:
        valid_mask = (j <= i) & ((i - j) <= window_size)

    scores = scores.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn_probs = torch.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_probs, v)

    # 8. Merge heads back to (B, S, D) and apply output projection
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
    proj_out = F.linear(attn_out, w["output_projection"])

    # 9. Sandwich RMSNorm (post-attention norm before residual addition)
    normed_out = rms_norm(proj_out, w["post_attention_gain"], eps)

    return (x + normed_out).to(torch.float64)
