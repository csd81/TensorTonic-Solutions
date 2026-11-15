import torch
import torch.nn.functional as F

def glm_swiglu(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
) -> torch.Tensor:
    """
    Returns a float64 tensor with shape (n_tokens, hidden_size).
    """
    gate = F.silu(x @ W_gate)
    up = x @ W_up
    return (gate * up) @ W_down
