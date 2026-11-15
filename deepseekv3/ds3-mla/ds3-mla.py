import math
import torch
import torch.nn.functional as F

def apply_rope(t: torch.Tensor, cos_freq: torch.Tensor, sin_freq: torch.Tensor) -> torch.Tensor:
    d = t.shape[-1]
    t1 = t[..., : d // 2]
    t2 = t[..., d // 2 :]

    # Align cos_freq and sin_freq with (B, num_heads, S, d // 2)
    while cos_freq.ndim < t1.ndim:
        cos_freq = cos_freq.unsqueeze(0)
        sin_freq = sin_freq.unsqueeze(0)

    out1 = t1 * cos_freq - t2 * sin_freq
    out2 = t1 * sin_freq + t2 * cos_freq
    return torch.cat([out1, out2], dim=-1)


def mla(
    x: torch.Tensor,
    W_dkv: torch.Tensor,
    W_uk: torch.Tensor,
    W_uv: torch.Tensor,
    W_dq: torch.Tensor,
    W_uq: torch.Tensor,
    W_qr: torch.Tensor,
    W_kr: torch.Tensor,
    W_o: torch.Tensor,
    cos_freq: torch.Tensor,
    sin_freq: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """
    Returns MLA attention outputs with shape (batch, sequence, model_width).
    """
    B, S, _ = x.shape

    # 1. KV low-rank compression and per-head reconstruction
    c_kv = F.linear(x, W_dkv)  # (B, S, d_ckv)
    k_c = F.linear(c_kv, W_uk).view(B, S, num_heads, -1).transpose(1, 2)  # (B, H, S, d_c)
    v = F.linear(c_kv, W_uv).view(B, S, num_heads, -1).transpose(1, 2)    # (B, H, S, d_v)

    # 2. Query low-rank compression and per-head content + rotary reconstruction
    c_q = F.linear(x, W_dq)  # (B, S, d_cq)
    q_c = F.linear(c_q, W_uq).view(B, S, num_heads, -1).transpose(1, 2)  # (B, H, S, d_c)
    q_r = F.linear(c_q, W_qr).view(B, S, num_heads, -1).transpose(1, 2)  # (B, H, S, d_r)

    # 3. Shared rotary key projection
    k_r = F.linear(x, W_kr).unsqueeze(1)  # (B, 1, S, d_r)

    # 4. Apply RoPE exclusively to rotary components and broadcast shared k_r across heads
    q_r = apply_rope(q_r, cos_freq, sin_freq)
    k_r = apply_rope(k_r, cos_freq, sin_freq).expand(-1, num_heads, -1, -1)

    # 5. Concatenate content and rotary parts
    q = torch.cat([q_c, q_r], dim=-1)  # (B, H, S, d_c + d_r)
    k = torch.cat([k_c, k_r], dim=-1)  # (B, H, S, d_c + d_r)

    # 6. Scaled causal self-attention
    head_dim = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    causal_mask = torch.triu(torch.ones((S, S), dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)

    # 7. Aggregate values and project output
    context = (attn @ v).transpose(1, 2).contiguous().view(B, S, -1)
    return F.linear(context, W_o)
