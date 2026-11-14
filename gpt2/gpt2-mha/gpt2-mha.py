import math
import torch

def multi_head_attention(
    x: torch.Tensor,
    W_q: torch.Tensor,
    b_q: torch.Tensor,
    W_k: torch.Tensor,
    b_k: torch.Tensor,
    W_v: torch.Tensor,
    b_v: torch.Tensor,
    W_o: torch.Tensor,
    b_o: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """
    Returns causal multi-head attention outputs with the same shape as x.
    """
    batch, sequence, width = x.shape
    head_width = width // num_heads

    def split(projected):
        return projected.reshape(batch, sequence, num_heads, head_width).transpose(1, 2)

    # 1. Linear projections with transpose (.T) and biases: (B, S, D)
    Q = split(x @ W_q.T + b_q)
    K = split(x @ W_k.T + b_k)
    V = split(x @ W_v.T + b_v)

    # 3. Scaled dot-product attention scores: (B, num_heads, S, S)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(head_width)

    # 4. Apply causal mask (prevent attending to future tokens j > i)
    mask = torch.triu(torch.ones(sequence, sequence, dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))

    # 5. Softmax & weighted sum of values: (B, num_heads, S, head_width)
    heads = torch.softmax(scores, dim=-1) @ V

    # 6 & 7. Concatenate heads and output projection with bias: (B, S, D)
    out = heads.transpose(1, 2).reshape(batch, sequence, width) @ W_o.T + b_o

    return out.to(torch.float64)
