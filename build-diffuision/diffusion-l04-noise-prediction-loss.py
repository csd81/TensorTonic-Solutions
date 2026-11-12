import torch

def noise_prediction_loss(
    x0: torch.Tensor, 
    timesteps: torch.Tensor, 
    noise: torch.Tensor, 
    alpha_bars: torch.Tensor, 
    weight: torch.Tensor, 
    time_weight: torch.Tensor, 
    bias: torch.Tensor
) -> dict:
    """
    Returns a dict with the batch loss as a Python float under "loss" and the per-sample losses of shape (B,) under "per_sample_loss".
    """
    B = x0.shape[0]
    T = alpha_bars.shape[0]
    
    # Flatten x0 and noise in row-major order to shape (B, D)
    x0_flat = x0.reshape(B, -1)
    noise_flat = noise.reshape(B, -1)
    
    # Index alpha_bars for each sample and reshape to (B, 1) for broadcasting
    a_bar = alpha_bars[timesteps].unsqueeze(-1)
    
    # Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    x_t = torch.sqrt(a_bar) * x0_flat + torch.sqrt(1.0 - a_bar) * noise_flat
    
    # Normalize timesteps (t / T) with matching dtype and shape (B, 1)
    t_norm = timesteps.to(x0.dtype).unsqueeze(-1) / T
    
    # Linear denoiser prediction: eps_hat = x_t @ W + (t / T) * w_t + b
    eps_hat = x_t @ weight + t_norm * time_weight.unsqueeze(0) + bias.unsqueeze(0)
    
    # Mean squared error averaged over feature dimension D -> shape (B,)
    per_sample_loss = ((eps_hat - noise_flat) ** 2).mean(dim=-1)
    
    # Batch loss averaged over B -> Python float
    loss = per_sample_loss.mean().item()
    
    return {
        "loss": loss,
        "per_sample_loss": per_sample_loss
    }
