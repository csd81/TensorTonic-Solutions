import math
import torch


def causal_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
  """
    Returns a float32 tensor of shape (batch, seq_q, d_v).
    """
  seq_q = Q.size(1)
  seq_k = K.size(1)
  d_k = Q.size(-1)

  # 1. Compute scaled dot-product attention scores: (batch, seq_q, seq_k)
  scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

  # 2. Create a boolean causal mask where True indicates j > i (future positions)
  # torch.triu with diagonal=1 sets elements strictly above the main diagonal to True
  mask = torch.ones((seq_q, seq_k), dtype=torch.bool, device=Q.device).triu(
      diagonal=1
  )

  # 3. Mask out future positions with -inf before applying softmax
  scores = scores.masked_fill(mask, float("-inf"))
  weights = torch.softmax(scores, dim=-1)

  # 4. Multiply attention weights by V: (batch, seq_q, d_v)
  return torch.matmul(weights, V).to(torch.float32)
