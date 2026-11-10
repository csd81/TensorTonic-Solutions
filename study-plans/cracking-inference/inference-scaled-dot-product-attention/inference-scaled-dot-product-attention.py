import math
import torch

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Returns a float32 attention tensor with shape (batch, query length, value width).
    """
    # Get the feature dimension d_k from the last axis of the query/key
    d_k = query.size(-1)
    
    # 1. Compute the dot products between queries and keys, then scale
    # key.transpose(-2, -1) swaps the sequence length and feature dimension of the keys
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. Apply the mask if it is provided
    # Positions where mask is True are blocked by setting them to negative infinity
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
        
    # 3. Apply softmax over the last dimension (key positions) to get attention weights
    attention_weights = torch.softmax(scores, dim=-1)
    
    # 4. Multiply attention weights by the values
    output = torch.matmul(attention_weights, value)
    
    return output
