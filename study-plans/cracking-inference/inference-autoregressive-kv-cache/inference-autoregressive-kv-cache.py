import math
import torch

def cached_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple:
    """
    Returns (outputs, key_cache, value_cache), float32 tensors in sequence order.
    """
    B, S, dk = query.shape
    dv = value.shape[-1]
    
    outputs = []
    
    # Initialize incremental key and value caches
    key_cache = torch.empty((B, 0, dk), dtype=query.dtype, device=query.device)
    value_cache = torch.empty((B, 0, dv), dtype=value.dtype, device=value.device)
    
    # Simulate step-by-step token-by-token generation
    for t in range(S):
        # Extract the current token's Q, K, V components
        q_t = query[:, t:t+1, :]
        k_t = key[:, t:t+1, :]
        v_t = value[:, t:t+1, :]
        
        # Incrementally append the new token's key and value to the cache
        key_cache = torch.cat([key_cache, k_t], dim=1)
        value_cache = torch.cat([value_cache, v_t], dim=1)
        
        # Calculate attention scores against all cached tokens up to step t
        # Shape: (B, 1, t+1)
        scores = torch.matmul(q_t, key_cache.transpose(-2, -1)) / math.sqrt(dk)
        
        # Apply softmax over the sequence length dimension (dim=-1)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Compute the weighted sum over cached values
        # Shape: (B, 1, dv)
        o_t = torch.matmul(attn_weights, value_cache)
        outputs.append(o_t)
        
    # Concatenate per-step outputs along the sequence dimension
    outputs = torch.cat(outputs, dim=1)
    
    return outputs, key_cache, value_cache
