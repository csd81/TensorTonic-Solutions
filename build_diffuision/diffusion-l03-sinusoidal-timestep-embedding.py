import math
import torch

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    Returns the float32 sinusoidal embedding of shape (B, dim).
    """
    half = dim // 2
    
    # Create frequency indices i = [0, 1, ..., half - 1] on the same device as timesteps
    i = torch.arange(half, dtype=torch.float32, device=timesteps.device)
    
    # Compute f_i = exp(-ln(max_period) * i / half)
    freqs = torch.exp(-math.log(max_period) * i / half)
    
    # Broadcast timesteps (B, 1) with freqs (1, half) to get angles a_{b, i} of shape (B, half)
    args = timesteps[:, None].to(torch.float32) * freqs[None, :]
    
    # Concatenate cosines followed by sines along the feature dimension -> (B, 2 * half)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    # If dim is odd, append a column of zeros -> (B, dim)
    if dim % 2 == 1:
        zeros = torch.zeros((timesteps.shape[0], 1), dtype=torch.float32, device=timesteps.device)
        embedding = torch.cat([embedding, zeros], dim=-1)
        
    return embedding
