import math
import torch

def sandwich_norm(
    x: torch.Tensor,
    sublayer_output: torch.Tensor,
    post_gain: torch.Tensor,
    num_layers: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Returns the depth-scaled sandwich-norm residual output.
    """
    x = x.to(torch.float64)
    s = sublayer_output.to(torch.float64)
    gamma = post_gain.to(torch.float64)

    # Compute RMS along the hidden dimension D (last dimension)
    rms = torch.sqrt(torch.mean(s.pow(2), dim=-1, keepdim=True) + eps)

    # Normalize, apply learned post-gain, and scale by 1 / sqrt(num_layers)
    z = (s / rms) * gamma / math.sqrt(num_layers)

    # Add back to the residual stream
    y = x + z
    return y.to(torch.float64)
