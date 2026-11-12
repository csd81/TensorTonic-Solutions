import torch

def causal_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Returns the attention tensor with the same shape and dtype as q.
    """
    B, H_q, S, D = q.shape
    _, H_kv, _, _ = k.shape
    G = H_q // H_kv
    
    # Perform all intermediate operations in float32 for numerical stability
    q_32 = q.to(torch.float32)
    k_32 = k.to(torch.float32)
    v_32 = v.to(torch.float32)
    
    # Reshape q to group query heads: (B, H_kv, G, S, D)
    q_grouped = q_32.view(B, H_kv, G, S, D)
    
    # Compute attention scores using Einstein summation to avoid head duplication: (B, H_kv, G, S, T)
    scores = torch.einsum('bhgsd,bhtd->bhgst', q_grouped, k_32) / (D ** 0.5)
    
    # Construct and apply a causal upper-triangular mask
    mask = torch.triu(torch.full((S, S), float('-inf'), device=q.device), diagonal=1)
    scores = scores + mask  # Implicitly broadcasts across batch, head, and group dimensions
    
    # Compute attention probabilities
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Linearly combine the values: (B, H_kv, G, S, D)
    out = torch.einsum('bhgst,bhtd->bhgsd', attn_weights, v_32)
    
    # Merge grouped head dimensions back into the original shape
    out = out.view(B, H_q, S, D)
    
    return out.to(q.dtype)
