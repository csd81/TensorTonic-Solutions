import torch

def nope_attention(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    gamma_q: torch.Tensor,
    gamma_k: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Returns NoPE grouped-query attention outputs with shape (batch, sequence, model_width).
    """
    x = x.to(torch.float64)
    W_q = W_q.to(torch.float64)
    W_k = W_k.to(torch.float64)
    W_v = W_v.to(torch.float64)
    W_o = W_o.to(torch.float64)
    gamma_q = gamma_q.to(torch.float64)
    gamma_k = gamma_k.to(torch.float64)

    B, S, D = x.shape
    d_h = D // num_query_heads
    num_groups = num_query_heads // num_kv_heads

    # 1. Linear projections (note: W shapes are (out_features, in_features))
    q = x @ W_q.T  # (B, S, H_q * d_h)
    k = x @ W_k.T  # (B, S, H_kv * d_h)
    v = x @ W_v.T  # (B, S, H_kv * d_h)

    # Reshape to separate heads and transpose to (B, H, S, d_h)
    q = q.view(B, S, num_query_heads, d_h).transpose(1, 2)
    k = k.view(B, S, num_kv_heads, d_h).transpose(1, 2)
    v = v.view(B, S, num_kv_heads, d_h).transpose(1, 2)

    # 2. Apply RMSNorm independently per head on Q and K
    q_rms = torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + eps)
    q = q * q_rms * gamma_q

    k_rms = torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + eps)
    k = k * k_rms * gamma_k

    # 3. Expand K and V heads for Grouped-Query Attention (GQA)
    # Maps query head h to kv head floor(h * H_kv / H_q)
    if num_groups > 1:
        k = k.repeat_interleave(num_groups, dim=1)
        v = v.repeat_interleave(num_groups, dim=1)

    # 4. Scaled causal dot-product attention (no RoPE / positional embeddings)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_h ** 0.5)
    causal_mask = torch.triu(
        torch.ones((S, S), dtype=torch.bool, device=x.device), diagonal=1
    )
    scores = scores.masked_fill(causal_mask, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v)  # (B, H_q, S, d_h)

    # 5. Concatenate query heads and apply output projection
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
    output = attn_out @ W_o.T

    return output.to(torch.float64)
