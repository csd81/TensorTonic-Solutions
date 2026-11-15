import math
import torch

def _rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    return x * gamma * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    return torch.cat((-x[..., d // 2 :], x[..., : d // 2]), dim=-1)

def _apply_partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    if rope_dim == 0:
        return x
    x_rot, x_pass = x[..., :rope_dim], x[..., rope_dim:]
    cos_f = torch.cat([cos, cos], dim=-1).unsqueeze(0)
    sin_f = torch.cat([sin, sin], dim=-1).unsqueeze(0)
    x_rot_out = x_rot * cos_f + _rotate_half(x_rot) * sin_f
    return torch.cat((x_rot_out, x_pass), dim=-1)

def glm_attention(
    x: torch.Tensor,
    W_qkv: torch.Tensor,
    b_qkv: torch.Tensor,
    W_o: torch.Tensor,
    qk_gamma_q: torch.Tensor,
    qk_gamma_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_dim: int,
    eps_rms: float = 1e-5,
) -> torch.Tensor:
    n_tokens = x.shape[0]
    q_size = num_attention_heads * head_dim
    kv_size = num_kv_heads * head_dim

    # 1. Fused QKV Projection & Split
    qkv = x @ W_qkv + b_qkv
    q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

    # Reshape to (num_heads, n_tokens, head_dim)
    q = q.view(n_tokens, num_attention_heads, head_dim).transpose(0, 1)
    k = k.view(n_tokens, num_kv_heads, head_dim).transpose(0, 1)
    v = v.view(n_tokens, num_kv_heads, head_dim).transpose(0, 1)

    # 2. Independent QK-Norm (applied before RoPE)
    q = _rms_norm(q, qk_gamma_q, eps_rms)
    k = _rms_norm(k, qk_gamma_k, eps_rms)

    # 3. Partial RoPE
    q = _apply_partial_rope(q, cos, sin, rope_dim)
    k = _apply_partial_rope(k, cos, sin, rope_dim)

    # 4. GQA Head Expansion
    q_mult = num_attention_heads // num_kv_heads
    if q_mult > 1:
        k = k.repeat_interleave(q_mult, dim=0)
        v = v.repeat_interleave(q_mult, dim=0)

    # 5. Causal Scaled Dot-Product Attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    mask = torch.triu(torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    attn_weights = torch.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v)

    # 6. Merge Heads & Output Projection
    attn_out = attn_out.transpose(0, 1).contiguous().view(n_tokens, q_size)
    return attn_out @ W_o
