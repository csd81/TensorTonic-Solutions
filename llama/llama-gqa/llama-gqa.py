import math
import torch
import torch.nn.functional as F

def grouped_query_attention(
    x: torch.Tensor, 
    weights: dict[str, torch.Tensor], 
    num_query_heads: int, 
    num_kv_heads: int
) -> torch.Tensor:
    """
    Returns the float64 grouped-query attention output.
    """
    # Ensure input and weights are float64
    x = x.to(torch.float64)
    w_q = weights["query_projection"].to(torch.float64)
    w_k = weights["key_projection"].to(torch.float64)
    w_v = weights["value_projection"].to(torch.float64)
    w_o = weights["output_projection"].to(torch.float64)

    B, S, D = x.shape
    head_dim = D // num_query_heads
    num_groups = num_query_heads // num_kv_heads

    # 1. Linear projections (weights are stored as [out_features, in_features])
    q = F.linear(x, w_q)  # (B, S, D)
    k = F.linear(x, w_k)  # (B, S, num_kv_heads * head_dim)
    v = F.linear(x, w_v)  # (B, S, num_kv_heads * head_dim)

    # 2. Reshape and transpose to separate heads: (B, num_heads, S, head_dim)
    q = q.view(B, S, num_query_heads, head_dim).transpose(1, 2)
    k = k.view(B, S, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(B, S, num_kv_heads, head_dim).transpose(1, 2)

    # 3. Repeat KV heads to match the number of query heads
    # repeat_interleave repeats each head `num_groups` times consecutively
    if num_groups > 1:
        k = k.repeat_interleave(num_groups, dim=1)  # (B, num_query_heads, S, head_dim)
        v = v.repeat_interleave(num_groups, dim=1)  # (B, num_query_heads, S, head_dim)

    # 4. Scaled dot-product attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    # 5. Apply causal mask (mask future positions above the main diagonal)
    causal_mask = torch.triu(
        torch.full((S, S), float("-inf"), dtype=torch.float64, device=x.device),
        diagonal=1
    )
    scores = scores + causal_mask

    attn_weights = F.softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v)  # (B, num_query_heads, S, head_dim)

    # 6. Concatenate query heads back to (B, S, D) and apply output projection
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
    output = F.linear(attn_output, w_o)

    return output.to(torch.float64)
