import math
import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns non-causal attention outputs.
    """
    # Ensure inputs are float64 as required by constraints
    Q = Q.to(torch.float64)
    K = K.to(torch.float64)
    V = V.to(torch.float64)
    
    # d_k is the final dimension (key width) of Q and K
    d_k = Q.size(-1)
    
    # 1. Compute query-key dot products: (B, S_q, d_k) @ (B, d_k, S_k) -> (B, S_q, S_k)
    # 2. Scale scores by sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 3. Apply softmax across the key positions (last dimension)
    attention_weights = torch.softmax(scores, dim=-1)
    
    # 4. Multiply attention weights by V: (B, S_q, S_k) @ (B, S_k, d_v) -> (B, S_q, d_v)
    output = torch.matmul(attention_weights, V)
    
    return output
