import torch

def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Returns a float64 tensor with the same shape as x.
    """
    # Ensure inputs are float64 for high precision and output requirements
    x = x.to(torch.float64)
    gamma = gamma.to(torch.float64)
    
    # Compute the mean of squares along the final dimension, preserving dimensions for broadcasting
    mean_square = torch.mean(x ** 2, dim=-1, keepdim=True)
    
    # Compute the reciprocal root mean square (1 / sqrt(mean(x^2) + eps))
    rms = torch.rsqrt(mean_square + eps)
    
    # Normalize x and scale by learned gain gamma
    return gamma * (x * rms)
