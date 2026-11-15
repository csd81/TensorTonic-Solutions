import torch

def rope_freqs(n_tokens: int, rope_dim: int, rope_theta: float) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing float64 cos and sin tables.
    """
    # Even indices 0, 2, 4, ..., rope_dim - 2 corresponding to 2i
    pair_indices = torch.arange(0, rope_dim, 2, dtype=torch.float64)
    
    # theta_i = 1 / (rope_theta ** (2i / rope_dim))
    inv_freq = 1.0 / (rope_theta ** (pair_indices / rope_dim))
    
    # Zero-based token positions: 0, 1, ..., n_tokens - 1
    positions = torch.arange(n_tokens, dtype=torch.float64)
    
    # Outer product to get angles a_{p, i} = p * theta_i of shape (n_tokens, rope_dim // 2)
    angles = torch.outer(positions, inv_freq)
    
    return {
        "cos": torch.cos(angles),
        "sin": torch.sin(angles),
    }
