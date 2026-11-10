import torch

def apply_rotary_position_embeddings(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
) -> tuple:
    """
    Applies rotary position embeddings (RoPE) to queries and keys tensors 
    using adjacent feature pairs and absolute positions.
    
    Shapes:
        query, key: (B, H, S, d)
        positions: (S,)
    Returns:
        (rotated_query, rotated_key)
    """
    B, H, S, d = query.shape
    device = query.device
    
    # Compute inverse frequencies (theta) for each feature pair
    i = torch.arange(0, d // 2, dtype=torch.float32, device=device)
    theta = base ** (-2 * i / d)
    
    # Compute the product of position and theta: shape (S, d // 2)
    p = positions.to(torch.float32)
    p_theta = p.unsqueeze(-1) * theta.unsqueeze(0)
    
    # Compute cos and sin, reshaped for broadcasting: (1, 1, S, d // 2)
    cos = torch.cos(p_theta).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(p_theta).unsqueeze(0).unsqueeze(0)
    
    # Apply rotation to adjacent even-odd feature pairs for query
    q_even, q_odd = query[..., 0::2], query[..., 1::2]
    q_rot_even = q_even * cos - q_odd * sin
    q_rot_odd = q_even * sin + q_odd * cos
    
    rotated_query = torch.empty_like(query)
    rotated_query[..., 0::2] = q_rot_even
    rotated_query[..., 1::2] = q_rot_odd
    
    # Apply rotation to adjacent even-odd feature pairs for key
    k_even, k_odd = key[..., 0::2], key[..., 1::2]
    k_rot_even = k_even * cos - k_odd * sin
    k_rot_odd = k_even * sin + k_odd * cos
    
    rotated_key = torch.empty_like(key)
    rotated_key[..., 0::2] = k_rot_even
    rotated_key[..., 1::2] = k_rot_odd
    
    return rotated_query, rotated_key
