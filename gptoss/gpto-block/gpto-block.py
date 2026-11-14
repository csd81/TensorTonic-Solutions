import math
import torch
import torch.nn.functional as F


def _rms_norm(x: torch.Tensor, g: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x * rms * g


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (n, ..., head_dim), cos/sin: (n, head_dim // 2)
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


def _gpto_swiglu(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1.0)


def gpto_block(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    weights: dict,
    top_k: int,
    q_mult: int,
    head_dim: int,
    num_kv_heads: int,
    sliding_window: int,
    layer_idx: int,
    swiglu_limit: float = 7.0,
    eps_rms: float = 1e-5,
) -> torch.Tensor:
    """
    Returns the transformed residual stream with the same shape as x.
    """
    n_tokens, hidden_size = x.shape
    num_q_heads = num_kv_heads * q_mult

    # 1. Pre-Attention RMSNorm
    t = _rms_norm(x, weights["norm1_g"], eps_rms)

    # 2. Fused QKV Projection & Split
    qkv = F.linear(t, weights["qkv_W"], weights["qkv_b"])
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    q = qkv[:, :q_dim].view(n_tokens, num_kv_heads, q_mult, head_dim)
    k = qkv[:, q_dim : q_dim + kv_dim].view(n_tokens, num_kv_heads, head_dim)
    v = qkv[:, q_dim + kv_dim :].view(n_tokens, num_kv_heads, head_dim)

    # 3. Rotary Position Embeddings (RoPE)
    q = _apply_rotary_emb(q, cos, sin)
    k = _apply_rotary_emb(k, cos, sin)

    # 4. Grouped Query Attention with Learned Sinks & Alternating Sliding Window
    k_exp = k[:, :, None, :].expand(-1, -1, q_mult, -1)
    v_exp = v[:, :, None, :].expand(-1, -1, q_mult, -1)
    sinks = weights["sinks"].reshape(num_kv_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)

    # Causal upper-triangular mask
    mask = torch.triu(x.new_full((n_tokens, n_tokens), float("-inf")), diagonal=1)
    # Apply banded sliding-window mask on even layers only
    if layer_idx % 2 == 0 and sliding_window > 0:
        mask = mask + torch.tril(
            x.new_full((n_tokens, n_tokens), float("-inf")), diagonal=-sliding_window
        )

    sm_scale = 1.0 / math.sqrt(head_dim)
    qk = torch.einsum("qhmd,khmd->hmqk", q, k_exp) * sm_scale
    qk = qk + mask[None, None, :, :]
    qk = torch.cat([qk, sinks], dim=-1)

    attn_weights = torch.softmax(qk, dim=-1)[..., :-1]
    attn_out = torch.einsum("hmqk,khmd->qhmd", attn_weights, v_exp).reshape(n_tokens, q_dim)

    # 5. Output Projection & First Residual Skip
    h = x + F.linear(attn_out, weights["out_W"], weights["out_b"])

    # 6. Pre-MoE RMSNorm
    t_moe = _rms_norm(h, weights["norm2_g"], eps_rms)

    # 7. Top-k MoE Router
    router_logits = F.linear(t_moe, weights["gate_W"], weights["gate_b"])
    topk = torch.topk(router_logits, k=top_k, dim=-1, sorted=True)
    expert_weights = torch.softmax(topk.values, dim=-1)
    expert_indices = topk.indices

    # 8. Sparse Expert Forward (Clamped SwiGLU)
    w1 = weights["W1"][expert_indices]  # (n, top_k, 2I, H)
    b1 = weights["b1"][expert_indices]  # (n, top_k, 2I)
    w2 = weights["W2"][expert_indices]  # (n, top_k, H, I)
    b2 = weights["b2"][expert_indices]  # (n, top_k, H)

    expert_hidden = torch.einsum("bekh,bh->bek", w1, t_moe) + b1
    expert_act = _gpto_swiglu(expert_hidden, limit=swiglu_limit)
    expert_out = torch.einsum("behk,bek->beh", w2, expert_act) + b2

    # 9. Weighted Expert Combination & Second Residual Skip
    moe_out = torch.einsum("beh,be->bh", expert_out, expert_weights)
    return (h + moe_out).to(torch.float64)
