import torch

def ddpm_step(x_t: torch.Tensor, t: int, predicted_noise: torch.Tensor, betas: torch.Tensor, noise: torch.Tensor, variance_type: str) -> torch.Tensor:
    """
    Returns the sample x_{t-1} after one DDPM reverse step, with the shape and dtype of x_t.
    """
    betas = betas.to(device=x_t.device, dtype=x_t.dtype)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bars[t]

    # Compute the reverse mean mu
    coeff = beta_t / torch.sqrt(1.0 - alpha_bar_t)
    mu = (1.0 / torch.sqrt(alpha_t)) * (x_t - coeff * predicted_noise)

    # At t == 0, return the mean without adding noise
    if t == 0:
        return mu

    # Determine variance sigma_t^2 based on variance_type
    if variance_type == "fixed_large":
        variance = beta_t
    elif variance_type == "fixed_small":
        alpha_bar_prev = alpha_bars[t - 1]
        variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
    else:
        raise ValueError(f"Unsupported variance_type: {variance_type}")

    sigma_t = torch.sqrt(variance)
    return mu + sigma_t * noise
