import math
import torch


def sliding_window_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, window_size: int
) -> torch.Tensor:
    """Returns a float64 attention output tensor."""
    Q = Q.to(torch.float64)
    K = K.to(torch.float64)
    V = V.to(torch.float64)

    _, _, seq_len, d_h = Q.shape

    # 1. Compute scaled dot-product scores: (B, H, S, S)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_h)

    # 2. Build sliding window causal mask
    # Query index i (column vector), Key index j (row vector)
    i = torch.arange(seq_len, device=Q.device).unsqueeze(1)
    j = torch.arange(seq_len, device=Q.device).unsqueeze(0)

    # Visible when j <= i (causal) and (i - j) <= window_size
    valid_mask = (j <= i) & ((i - j) <= window_size)

    # 3. Mask out invalid positions with -inf before softmax
    scores = scores.masked_fill(~valid_mask, float("-inf"))

    # 4. Compute attention weights and weighted sum of values
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output.to(torch.float64)
