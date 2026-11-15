import torch

def partial_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing q_rotated and k_rotated tensors.
    """
    if rope_dim == 0:
        return {
            "q_rotated": q.clone(),
            "k_rotated": k.clone(),
        }

    # Repeat cos and sin across the two contiguous halves -> (n_tokens, rope_dim)
    # Unsqueeze head dim for broadcasting against (n_tokens, n_heads, rope_dim)
    cos_full = torch.cat([cos, cos], dim=-1).unsqueeze(1)
    sin_full = torch.cat([sin, sin], dim=-1).unsqueeze(1)

    def apply_rotary(x: torch.Tensor) -> torch.Tensor:
        x_rot = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]

        # Split leading rotary prefix into two contiguous halves
        half = rope_dim // 2
        a = x_rot[..., :half]
        b = x_rot[..., half:]
        rotated_half = torch.cat([-b, a], dim=-1)

        x_rot_out = x_rot * cos_full + rotated_half * sin_full
        return torch.cat([x_rot_out, x_pass], dim=-1)

    return {
        "q_rotated": apply_rotary(q),
        "k_rotated": apply_rotary(k),
    }
