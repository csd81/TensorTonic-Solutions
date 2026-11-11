import torch

def min_snr_weighted_loss(
    per_sample_mse: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bars: torch.Tensor,
    gamma: float,
    prediction_type: str,
) -> dict:
    """
    Returns a dict with the per-sample "snr" and "weights" of shape (B,) and the weighted batch "loss" as a Python float.
    """
    if prediction_type not in ("epsilon", "x0", "v"):
        raise ValueError(f"Unsupported prediction_type: {prediction_type!r}. Expected 'epsilon', 'x0', or 'v'.")

    alpha_bars = alpha_bars.to(device=per_sample_mse.device, dtype=per_sample_mse.dtype)
    timesteps = timesteps.to(device=per_sample_mse.device)

    # 1. Gather per-sample alpha_bar_t and compute SNR
    alpha_bar_t = alpha_bars[timesteps]
    snr = alpha_bar_t / (1.0 - alpha_bar_t)

    # 2. Clamp SNR at gamma from above
    capped_snr = torch.clamp(snr, max=gamma)

    # 3. Compute weights based on prediction_type
    if prediction_type == "epsilon":
        weights = capped_snr / snr
    elif prediction_type == "x0":
        weights = capped_snr
    else:  # prediction_type == "v"
        weights = capped_snr / (snr + 1.0)

    # 4. Compute weighted mean loss as a Python float
    loss = (weights * per_sample_mse).mean().item()

    return {
        "snr": snr,
        "weights": weights,
        "loss": loss,
    }
