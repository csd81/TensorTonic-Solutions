import torch

def posterior_mean_variance(x0: torch.Tensor, x_t: torch.Tensor, timesteps: torch.Tensor, betas: torch.Tensor) -> dict:
    """
    Returns a dict with the posterior "mean" (same shape as x0) and the per-sample "variance" and "log_variance" of shape (B,).
    """
    # Ensure betas matches x0's dtype and device
    betas = betas.to(device=x0.device, dtype=x0.dtype)
    timesteps = timesteps.to(device=x0.device)
    
    # 1. Compute alphas and cumulative product alpha_bars
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    # Pad alpha_bars on the left with 1.0 so alpha_bars_prev[t] = alpha_bar_{t-1} (with alpha_bar_{-1} = 1.0)
    alpha_bars_prev = torch.cat([torch.ones(1, device=x0.device, dtype=x0.dtype), alpha_bars[:-1]], dim=0)
    
    # 2. Gather per-sample schedule values for the batch
    beta_t = betas[timesteps]
    alpha_t = alphas[timesteps]
    alpha_bar_t = alpha_bars[timesteps]
    alpha_bar_t_prev = alpha_bars_prev[timesteps]
    
    # 3. Compute posterior mean coefficients (shape: (B,))
    coef_x0 = (beta_t * torch.sqrt(alpha_bar_t_prev)) / (1.0 - alpha_bar_t)
    coef_xt = ((1.0 - alpha_bar_t_prev) * torch.sqrt(alpha_t)) / (1.0 - alpha_bar_t)
    
    # Reshape coefficients to (B, 1, 1, ...) to broadcast across all non-batch dimensions of x0
    broadcast_shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
    mean = coef_x0.view(broadcast_shape) * x0 + coef_xt.view(broadcast_shape) * x_t
    
    # 4. Compute posterior variance (shape: (B,))
    variance = ((1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)) * beta_t
    
    # 5. Compute clipped log variance: for t == 0, use the variance at t = 1 (betas[1] * (1 - alpha_bar_0) / (1 - alpha_bar_1))
    # Since alpha_bar_0 == alphas[0] == 1 - betas[0], (1 - alpha_bar_0) == betas[0]
    var_t1 = ((1.0 - alpha_bars_prev[1]) / (1.0 - alpha_bars[1])) * betas[1]
    clipped_variance = torch.where(timesteps == 0, var_t1, variance)
    log_variance = torch.log(clipped_variance)
    
    return {
        "mean": mean,
        "variance": variance,
        "log_variance": log_variance
    }
