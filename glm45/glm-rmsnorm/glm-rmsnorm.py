import torch

def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float64 tensor with the same shape as x.
    """
    # Ensure float64 precision for calculation
    x_fp64 = x.to(torch.float64)
    gamma_fp64 = gamma.to(torch.float64)
    
    # Compute root mean square along the last axis: sqrt(mean(x^2) + eps)
    rms = torch.sqrt(torch.mean(x_fp64 ** 2, dim=-1, keepdim=True) + eps)
    
    # Normalize and scale by learned per-channel gain (broadcasted over last axis)
    return (x_fp64 / rms) * gamma_fp64
