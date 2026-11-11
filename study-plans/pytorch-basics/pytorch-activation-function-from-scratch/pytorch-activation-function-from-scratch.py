import torch

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    if method == "relu":
        return torch.clamp(x, min=0)
        
    elif method == "sigmoid":
        return 1.0 / (1.0 + torch.exp(-x))
        
    elif method == "tanh":
        # Safe tanh: sgn(x) * (1 - exp(-2|x|)) / (1 + exp(-2|x|))
        magnitude = torch.abs(x)
        decay = torch.exp(-2.0 * magnitude)
        positive_tanh = (1.0 - decay) / (1.0 + decay)
        return torch.where(x >= 0, positive_tanh, -positive_tanh)
        
    elif method == "leaky_relu":
        return torch.where(x > 0, x, 0.01 * x)
        
    else:
        raise ValueError(f"Unknown activation method: {method}")
