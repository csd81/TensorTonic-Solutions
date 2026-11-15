import torch
import torch.nn.functional as F

def per_layer_embedding(
    input_embeddings: torch.Tensor,
    layer_output: torch.Tensor,
    p_weight: torch.Tensor,
    g_weight: torch.Tensor,
    u_weight: torch.Tensor,
    p_norm_scale: torch.Tensor,
    u_norm_scale: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Returns the layer output with the gated per-layer embedding update added.
    """
    if eps is None:
        eps = 1e-6

    def rms_norm(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (x / rms) * scale

    # Project input embeddings into per-layer space and normalize
    p_proj = F.linear(input_embeddings, p_weight)
    p_l = rms_norm(p_proj, p_norm_scale)

    # Project layer output and apply exact GELU gating
    g_proj = F.linear(layer_output, g_weight)
    g_l = F.gelu(g_proj, approximate="none")

    # Gate, project back to model width, and normalize
    u_proj = F.linear(g_l * p_l, u_weight)
    u_l = rms_norm(u_proj, u_norm_scale)

    # Residual addition
    return layer_output + u_l
