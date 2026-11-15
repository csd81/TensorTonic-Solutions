import torch

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary embeddings to tensor x along the last dimension.
    x: (..., d_r)
    cos: (..., d_r / 2)
    sin: (..., d_r / 2)
    """
    # Split x into pairs (x1, x2) along the last dimension
    d_r = x.shape[-1]
    x1 = x[..., :d_r // 2]
    x2 = x[..., d_r // 2:]
    
    # Apply rotation: [x1, x2] * [cos, -sin; sin, cos] = [x1*cos - x2*sin, x1*sin + x2*cos]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)

def decoupled_rope(
    k_content: torch.Tensor, 
    k_rope_input: torch.Tensor, 
    cos_freq: torch.Tensor, 
    sin_freq: torch.Tensor, 
    num_heads: int
) -> torch.Tensor:
    """
    Returns full MLA keys with shared rotary components.
    
    Args:
        k_content: Tensor of shape (B, H, S, d_c) - float64
        k_rope_input: Tensor of shape (B, S, d_r) - float64
        cos_freq: Tensor of shape (S, d_r / 2) - float64
        sin_freq: Tensor of shape (S, d_r / 2) - float64
        num_heads: int (equals H)
        
    Returns:
        Tensor of shape (B, H, S, d_c + d_r) - float64
    """
    # Ensure float64 as requested
    k_content = k_content.to(torch.float64)
    k_rope_input = k_rope_input.to(torch.float64)
    cos_freq = cos_freq.to(torch.float64)
    sin_freq = sin_freq.to(torch.float64)

    # 1. Apply RoPE to k_rope_input (shape: B, S, d_r)
    rotated_rope = apply_rotary_emb(k_rope_input, cos_freq, sin_freq)  # (B, S, d_r)

    # 2. Expand rotated key across num_heads (B, S, d_r) -> (B, 1, S, d_r) -> (B, H, S, d_r)
    rotated_rope_expanded = rotated_rope.unsqueeze(1).expand(-1, num_heads, -1, -1)

    # 3. Concatenate content keys first and rotary keys second: (B, H, S, d_c + d_r)
    k_full = torch.cat([k_content, rotated_rope_expanded], dim=-1)

    return k_full
