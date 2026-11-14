import math
import torch

def llada_bidirectional_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the attention output and the attention weights as a tuple.
    """
    d = queries.shape[-1]
    scale = 1.0 / math.sqrt(d)

    # 1. Compute scaled dot-product scores: (B, H, L, d) @ (B, H, d, L) -> (B, H, L, L)
    scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale

    # 2. Construct the additive padding bias B_{b, j}
    # Use torch.finfo to get the most negative finite value for the query dtype
    min_val = torch.finfo(queries.dtype).min
    
    # Expand attention_mask from (B, L) to (B, 1, 1, L) to broadcast across heads and query rows
    mask = attention_mask[:, None, None, :].to(device=queries.device, dtype=torch.bool)
    
    # Apply the bias without mutating the input tensors
    bias = torch.where(mask, torch.zeros_like(scores), torch.full_like(scores, min_val))
    scores = scores + bias

    # 3. Compute attention weights over the key dimension (-1)
    attn_weights = torch.softmax(scores, dim=-1)

    # 4. Compute attention output: (B, H, L, L) @ (B, H, L, d) -> (B, H, L, d)
    attn_output = torch.matmul(attn_weights, values)

    return attn_output, attn_weights
