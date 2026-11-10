import math
import torch

def grouped_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns an attention tensor with the same shape as hidden_states.
    """
    # 1. Validate head compatibility
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
        
    B, S, D = hidden_states.shape
    d = D // num_query_heads
    group_size = num_query_heads // num_kv_heads
    
    # 2. Project hidden states (features are along columns)
    # Shapes: Q -> (B, S, D), K/V -> (B, S, num_kv_heads * d)
    Q = torch.matmul(hidden_states, w_q)
    K = torch.matmul(hidden_states, w_k)
    V = torch.matmul(hidden_states, w_v)
    
    # 3. Reshape and transpose to prepare for attention
    # Shape: (B, num_heads, S, d)
    Q = Q.view(B, S, num_query_heads, d).transpose(1, 2)
    K = K.view(B, S, num_kv_heads, d).transpose(1, 2)
    V = V.view(B, S, num_kv_heads, d).transpose(1, 2)
    
    # 4. Expand K and V heads to match Q heads (Grouped-Query Sharing)
    # repeat_interleave repeats each head consecutively: [K0, K0, K1, K1...]
    # Shape: (B, num_query_heads, S, d)
    K = torch.repeat_interleave(K, repeats=group_size, dim=1)
    V = torch.repeat_interleave(V, repeats=group_size, dim=1)
    
    # 5. Compute scaled dot-product attention scores
    # Shape: (B, num_query_heads, S, S)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
    
    # 6. Apply causal mask if required
    if causal:
        mask = torch.full((S, S), float('-inf'), device=scores.device, dtype=scores.dtype)
        mask = torch.triu(mask, diagonal=1)
        scores = scores + mask
        
    # 7. Softmax and values aggregation
    attention_probs = torch.softmax(scores, dim=-1)
    # Shape: (B, num_query_heads, S, d)
    H = torch.matmul(attention_probs, V)
    
    # 8. Concatenate heads and apply output projection
    # Shape: (B, S, D)
    H = H.transpose(1, 2).contiguous().view(B, S, D)
    Y = torch.matmul(H, w_o)
    
    return Y
