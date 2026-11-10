import torch

def rotary_embed(
    q: torch.Tensor, k: torch.Tensor,
    positions: torch.Tensor, inv_freq: torch.Tensor,
) -> dict:
    """
    Applies rotary position embeddings to adjacent coordinate pairs in query and key tensors.
    All rotation arithmetic is performed in float32 to ensure precision, and results are 
    cast back to the original dtypes of q and k.
    """
    B, H, S, D = q.shape
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype

    # Reshape positions to be broadcastable over (B, 1, S, 1)
    if positions.dim() == 1:
        p = positions.view(1, 1, S, 1)
    else:
        p = positions.view(B, 1, S, 1)
        
    # Reshape inv_freq to be broadcastable over (1, 1, 1, D // 2)
    w = inv_freq.view(1, 1, 1, -1)
    
    # Compute the angles (theta) in float32
    angles = p.to(torch.float32) * w.to(torch.float32)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    # Reshape q and k into coordinate pairs of shape (B, H, S, D // 2, 2)
    q_pairs = q.view(B, H, S, D // 2, 2)
    k_pairs = k.view(B, H, S, D // 2, 2)
    
    # Separate the even (x_2i) and odd (x_2i+1) components and cast to float32
    q_even = q_pairs[..., 0].to(torch.float32)
    q_odd = q_pairs[..., 1].to(torch.float32)
    k_even = k_pairs[..., 0].to(torch.float32)
    k_odd = k_pairs[..., 1].to(torch.float32)
    
    # Apply the rotation transformations
    q_even_rot = q_even * cos - q_odd * sin
    q_odd_rot = q_even * sin + q_odd * cos
    
    k_even_rot = k_even * cos - k_odd * sin
    k_odd_rot = k_even * sin + k_odd * cos
    
    # Recombine the pairs and restore the original shape and dtypes
    q_rotated = torch.stack([q_even_rot, q_odd_rot], dim=-1).view(B, H, S, D).to(orig_q_dtype)
    k_rotated = torch.stack([k_even_rot, k_odd_rot], dim=-1).view(B, H, S, D).to(orig_k_dtype)
    
    return {
        "q_rotated": q_rotated,
        "k_rotated": k_rotated
    }
