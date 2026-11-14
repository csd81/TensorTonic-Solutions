import torch

def precompute_rope_freqs(
    max_seq_len: int, 
    head_dim: int, 
    base: float = 500000.0
) -> dict[str, torch.Tensor]:
    """
    Returns float64 cosine and sine tables in a dictionary.
    """
    # 1. Frequency-pair indices m = [0, 1, ..., head_dim // 2 - 1]
    pair_indices = torch.arange(0, head_dim // 2, dtype=torch.float64)
    
    # 2. Inverse frequencies: omega_m = 1 / (base ** (2m / head_dim))
    inv_freqs = 1.0 / (base ** (2.0 * pair_indices / head_dim))
    
    # 3. Token positions p = [0, 1, ..., max_seq_len - 1]
    positions = torch.arange(0, max_seq_len, dtype=torch.float64)
    
    # 4. Outer product to get angles theta_{p, m} = p * omega_m -> shape (L, head_dim // 2)
    angles = torch.outer(positions, inv_freqs)
    
    return {
        "cos": torch.cos(angles),
        "sin": torch.sin(angles),
    }
