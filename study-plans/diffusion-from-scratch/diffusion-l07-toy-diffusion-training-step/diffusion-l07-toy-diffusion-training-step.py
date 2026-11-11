import math
import torch
import torch.nn.functional as F

def toy_diffusion_training_step(x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor, alpha_bars: torch.Tensor, params: dict, embed_dim: int, learning_rate: float, max_grad_norm: float) -> dict:
    """
    Returns a dict with "loss" (float), "grad_norm" (float, before clipping) and "params" (dict of updated tensors).
    """
    device = x0.device
    dtype = x0.dtype
    B = x0.shape[0]

    # 1. Forward diffuse x0 to x_t using each sample's own timestep
    ab_t = alpha_bars[timesteps].unsqueeze(-1)  # (B, 1)
    x_t = torch.sqrt(ab_t) * x0 + torch.sqrt(1.0 - ab_t) * noise

    # 2. Build sinusoidal timestep embeddings e(t) of shape (B, embed_dim)
    half = embed_dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=dtype) / half
    )  # (half,)
    angles = timesteps.to(dtype=dtype).unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if embed_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((B, 1), device=device, dtype=dtype)], dim=-1)

    # 3. Clone weights with requires_grad=True so original inputs remain untouched
    keys = ["W1", "b1", "W2", "b2"]
    w = {k: params[k].detach().clone().requires_grad_(True) for k in keys}

    # 4. Forward pass and MSE loss over all B * D entries
    mlp_in = torch.cat([x_t, emb], dim=-1)
    hidden = F.silu(mlp_in @ w["W1"] + w["b1"])
    eps_hat = hidden @ w["W2"] + w["b2"]
    loss_tensor = ((eps_hat - noise) ** 2).mean()

    # 5. Backpropagate to compute gradients and global L2 norm before clipping
    grads = torch.autograd.grad(loss_tensor, [w[k] for k in keys])
    grad_norm_tensor = torch.sqrt(sum((g ** 2).sum() for g in grads))

    # 6. Apply global gradient norm clipping and SGD update
    clip_coef = torch.clamp(max_grad_norm / (grad_norm_tensor + 1e-6), max=1.0)
    updated_params = {}
    with torch.no_grad():
        for k, g in zip(keys, grads):
            updated_params[k] = w[k] - learning_rate * clip_coef * g

    return {
        "loss": float(loss_tensor.item()),
        "grad_norm": float(grad_norm_tensor.item()),
        "params": updated_params,
    }
