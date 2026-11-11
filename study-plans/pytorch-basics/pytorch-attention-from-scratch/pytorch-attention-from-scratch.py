import math
import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 attention tensor of shape (batch, seq_q, d_v).
    """
    d_k = Q.size(-1)
    
    # 1. Compute raw dot-product scores: (batch, seq_q, d_k) @ (batch, d_k, seq_k) -> (batch, seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # 2. Scale by the square root of d_k
    scaled_scores = scores / math.sqrt(d_k)
    
    # 3. Apply softmax across the keys dimension (last axis)
    attn_weights = torch.softmax(scaled_scores, dim=-1)
    
    # 4. Multiply attention weights by values: (batch, seq_q, seq_k) @ (batch, seq_k, d_v) -> (batch, seq_q, d_v)
    output = torch.matmul(attn_weights, V)
    
    return output.to(torch.float32)
