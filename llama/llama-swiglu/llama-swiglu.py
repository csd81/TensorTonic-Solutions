import torch
import torch.nn.functional as F

def swiglu_ffn(x: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Returns the float64 SwiGLU output tensor.
    """
    # Ensure input and weights are cast to float64
    x = x.to(torch.float64)
    w_gate = weights["gate_projection"].to(torch.float64)
    w_up = weights["up_projection"].to(torch.float64)
    w_down = weights["down_projection"].to(torch.float64)
    
    # 1. Compute SiLU gate: g = SiLU(x @ W_gate^T)
    gate = F.silu(F.linear(x, w_gate))
    
    # 2. Compute up projection: x @ W_up^T
    up = F.linear(x, w_up)
    
    # 3. Elementwise multiplication and down projection: (g * up) @ W_down^T
    hidden = gate * up
    y = F.linear(hidden, w_down)
    
    return y.to(torch.float64)
