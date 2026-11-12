import torch

def flow_matching_loss(x_data: torch.Tensor, x_noise: torch.Tensor, t: torch.Tensor, weight: torch.Tensor, time_weight: torch.Tensor, bias: torch.Tensor) -> dict:
    """
    Returns a dict with "x_t", "target", "per_sample_loss" (tensors) and "loss" (Python float).
    """
    # Expand t from (B,) to (B, 1) for broadcasting across feature dimension D
    t_col = t.unsqueeze(-1)

    # 1. Interpolate along the straight path from data (t=0) to noise (t=1)
    x_t = (1.0 - t_col) * x_data + t_col * x_noise

    # 2. Constant conditional velocity along the straight line
    target = x_noise - x_data

    # 3. Predict velocity v = x_t @ W + t * w_time + b
    pred_v = x_t @ weight + t_col * time_weight + bias

    # 4. Compute per-sample MSE across features D and scalar batch loss
    per_sample_loss = ((pred_v - target) ** 2).mean(dim=-1)
    loss = per_sample_loss.mean().item()

    return {
        "x_t": x_t,
        "target": target,
        "per_sample_loss": per_sample_loss,
        "loss": loss,
    }
