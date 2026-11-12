import math
import torch

def attention_scores(q: torch.Tensor, k: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Returns scores of shape (batch, heads, query_length, key_length).
    """
    # Extract structural dimensions
    B, Sq, D = q.shape
    _, Sk, _ = k.shape
    
    # Calculate individual head dimension width
    dh = D // num_heads
    
    # Reshape hidden dimension into (num_heads, head_width)
    # Using reshape instead of view to handle non-contiguous inputs safely
    q_reshaped = q.reshape(B, Sq, num_heads, dh)
    k_reshaped = k.reshape(B, Sk, num_heads, dh)
    
    # Perform batched dot product contraction across the head width (d)
    # b: batch, q: query_len, k: key_len, h: heads, d: head_width
    scores = torch.einsum('bqhd,bkhd->bhqk', q_reshaped, k_reshaped)
    
    # Apply scaled attention stabilization factor
    return scores / math.sqrt(dh)
