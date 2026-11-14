import math
import torch

def ffn(x: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """
    Returns feedforward outputs with the same shape as x.
    """
    x = x.to(torch.float64)
    W1 = W1.to(torch.float64)
    b1 = b1.to(torch.float64)
    W2 = W2.to(torch.float64)
    b2 = b2.to(torch.float64)

    # 1. Expand model width: (B, S, D) @ (D, I) + (I,) -> (B, S, I)
    hidden = x @ W1.T + b1

    # 2. Apply GPT-2 tanh GELU approximation elementwise
    c = math.sqrt(2.0 / math.pi)
    gelu_out = 0.5 * hidden * (1.0 + torch.tanh(c * (hidden + 0.044715 * torch.pow(hidden, 3))))

    # 3. Project back to model width: (B, S, I) @ (I, D) + (D,) -> (B, S, D)
    out = gelu_out @ W2.T + b2

    return out.to(torch.float64)
