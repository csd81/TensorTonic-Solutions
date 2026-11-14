import math
import torch

def scale_residual_weights(W: torch.Tensor, num_residual_layers: int) -> torch.Tensor:
    """
    Returns residual weights scaled for model depth.
    """
    return (W / math.sqrt(num_residual_layers)).to(torch.float64)
