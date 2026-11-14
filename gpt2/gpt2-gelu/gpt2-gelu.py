import math
import torch

def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Returns the elementwise GPT-2 GELU approximation.
    """
    x = x.to(torch.float64)
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
    return 0.5 * x * (1.0 + torch.tanh(inner))
