import torch

def convert_prediction(
    model_output: torch.Tensor,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bars: torch.Tensor,
    prediction_type: str,
) -> dict:
    """
    Returns a dict with the "x0", "epsilon" and "v" predictions, each with the shape and dtype of x_t.
    """
    if prediction_type not in ("epsilon", "x0", "v"):
        raise ValueError(f"Unsupported prediction_type: {prediction_type!r}. Expected 'epsilon', 'x0', or 'v'.")

    alpha_bars = alpha_bars.to(device=x_t.device, dtype=x_t.dtype)
    timesteps = timesteps.to(device=x_t.device)

    # Use x_t.shape[0] (or -1) for the batch size integer
    alpha_bar_t = alpha_bars[timesteps]
    broadcast_shape = (x_t.shape[0],) + (1,) * (x_t.ndim - 1)

    a = torch.sqrt(alpha_bar_t).view(broadcast_shape)
    b = torch.sqrt(1.0 - alpha_bar_t).view(broadcast_shape)

    out = model_output.clone()

    if prediction_type == "epsilon":
        eps = out
        x0 = (x_t - b * eps) / a
        v = a * eps - b * x0
    elif prediction_type == "x0":
        x0 = out
        eps = (x_t - a * x0) / b
        v = a * eps - b * x0
    else:  # prediction_type == "v"
        v = out
        x0 = a * x_t - b * v
        eps = b * x_t + a * v

    return {
        "x0": x0,
        "epsilon": eps,
        "v": v,
    }
