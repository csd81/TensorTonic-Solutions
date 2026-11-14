import torch

def layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns layer-normalized values with the same shape as x.
    """
    # Ensure float64 precision for calculations
    x = x.to(torch.float64)
    gamma = gamma.to(torch.float64)
    beta = beta.to(torch.float64)

    # Compute mean and population variance (unbiased=False) along the last dimension
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)

    # Normalize and apply affine scale (gamma) and shift (beta)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta
