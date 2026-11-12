import torch

def ddim_step(x_t: torch.Tensor, predicted_noise: torch.Tensor, t: int, t_prev: int, alpha_bars: torch.Tensor, eta: float, noise: torch.Tensor, clip_sample: bool) -> dict:
    """
    Returns a dict with "x_prev" and "pred_x0", each with the shape and dtype of x_t.
    """
    alpha_bars = alpha_bars.to(device=x_t.device, dtype=x_t.dtype)

    alpha_bar_t = alpha_bars[t]
    alpha_bar_prev = alpha_bars[t_prev] if t_prev >= 0 else torch.ones_like(alpha_bar_t)

    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

    # 1. Predict x_0 from x_t and predicted_noise
    pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

    # 2. Optionally clamp pred_x0 to [-1, 1] and recompute effective noise
    if clip_sample:
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        eps = (x_t - sqrt_alpha_bar_t * pred_x0) / sqrt_one_minus_alpha_bar_t
    else:
        eps = predicted_noise

    # 3. Compute stochastic variance sigma^2 and standard deviation sigma
    variance = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * (1.0 - alpha_bar_t / alpha_bar_prev)
    sigma = eta * torch.sqrt(variance)

    # 4. Compute direction pointing to x_t (clamping under the root at 0 for numerical stability)
    dir_xt_coeff = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=0.0))

    # 5. Assemble x_prev
    x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt_coeff * eps + sigma * noise

    return {
        "x_prev": x_prev,
        "pred_x0": pred_x0,
    }
