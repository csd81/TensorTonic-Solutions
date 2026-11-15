import torch
import torch.nn.functional as F

def shared_expert(
    x: torch.Tensor, 
    W_gate: torch.Tensor, 
    W_up: torch.Tensor, 
    W_down: torch.Tensor
) -> torch.Tensor:
    """
    Returns shared-expert outputs with the same shape as x.
    
    Args:
        x: Tensor of shape (B, S, D)
        W_gate: Tensor of shape (I, D) or (D, I) depending on convention,
                wait - check constraints: W_gate, W_up shape (I, D), W_down shape (D, I)
                Note: In standard linear layer x @ W.T or F.linear(x, W) where W is (out_features, in_features).
    """
    # Ensure float64 dtype as required
    x = x.to(torch.float64)
    W_gate = W_gate.to(torch.float64)
    W_up = W_up.to(torch.float64)
    W_down = W_down.to(torch.float64)

    # Gate and Up projections: y = (SiLU(x @ W_gate.T) ⊙ (x @ W_up.T)) @ W_down.T
    # Or using F.linear(x, W) which computes x @ W.T
    gate = F.linear(x, W_gate)  # shape: (B, S, I)
    up = F.linear(x, W_up)      # shape: (B, S, I)
    
    # SwiGLU activation and element-wise product
    intermediate = F.silu(gate) * up  # shape: (B, S, I)
    
    # Down projection back to model width D
    out = F.linear(intermediate, W_down)  # shape: (B, S, D)
    
    return out
