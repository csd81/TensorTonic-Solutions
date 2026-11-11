import torch

def forward_diffusion_sample(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bars: torch.Tensor,
    noise: torch.Tensor
) -> torch.Tensor:
    """
    Returns the noisy samples x_t with the same shape and dtype as x0.
    """
    # Gather the alpha_bar value for each sample in the batch: shape (B,)
    alpha_bar_t = alpha_bars.to(device=x0.device, dtype=x0.dtype)[timesteps]

    # Reshape to (B, 1, 1, ...) to broadcast across all non-batch dimensions
    broadcast_shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
    alpha_bar_t = alpha_bar_t.view(broadcast_shape)

    signal_scale = torch.sqrt(alpha_bar_t)
    noise_scale = torch.sqrt(1.0 - alpha_bar_t)

    return signal_scale * x0 + noise_scale * noise
