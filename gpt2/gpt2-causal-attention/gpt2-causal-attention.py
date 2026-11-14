import math
import torch

def causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns causal attention outputs.
    """
    d_k = Q.shape[-1]
    seq_len = Q.shape[-2]
    
    # 1. Compute scaled dot-product attention scores: (B, S, d_k) @ (B, d_k, S) -> (B, S, S)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. Create an upper-triangular boolean mask above the main diagonal (j > i)
    causal_mask = torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=Q.device), 
        diagonal=1
    )
    
    # 3. Fill future positions with negative infinity
    scores = scores.masked_fill(causal_mask, float('-inf'))
    
    # 4. Apply softmax along the last dimension (keys) and compute weighted sum of V
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    
    return output.to(torch.float64)
