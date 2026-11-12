import math
import torch

def fit_data_power_law(
    data_sizes: torch.Tensor,
    losses: torch.Tensor,
    target_sizes: torch.Tensor,
    floor_candidates: torch.Tensor,
) -> dict:
    """
    Returns a dict: floor, coefficient, exponent, predictions (float64 tensor), rmse.
    """
    d = data_sizes.to(torch.float64)
    l = losses.to(torch.float64)
    targets = target_sizes.to(torch.float64)

    log_d = torch.log(d)
    x_mean = log_d.mean()
    x_centered = log_d - x_mean
    denom = torch.sum(x_centered ** 2)

    min_loss = torch.min(l).item()

    best_rmse = float("inf")
    best_params = None

    for floor_tensor in floor_candidates:
        e = float(floor_tensor.item())

        # Candidate floor must be strictly below every observed loss
        if not (e < min_loss):
            continue

        y = torch.log(l - e)
        y_mean = y.mean()

        # Closed-form unweighted linear regression: y = a + b * log(D)
        b = torch.sum(x_centered * (y - y_mean)) / denom
        a = y_mean - b * x_mean

        alpha = float((-b).item())
        coeff = float(torch.exp(a).item())

        # Keep only fits with positive, finite coefficient A and exponent alpha
        if not (math.isfinite(coeff) and coeff > 0.0 and math.isfinite(alpha) and alpha > 0.0):
            continue

        # Evaluate RMSE on the original loss scale
        pred_obs = e + coeff * torch.pow(d, -alpha)
        rmse = float(torch.sqrt(torch.mean((pred_obs - l) ** 2)).item())

        if not math.isfinite(rmse):
            continue

        # Strict inequality (<) preserves candidate order on exact ties
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = (e, coeff, alpha, rmse)

    best_e, best_coeff, best_alpha, best_rmse = best_params
    predictions = (best_e + best_coeff * torch.pow(targets, -best_alpha)).to(
        dtype=torch.float64, device=target_sizes.device
    )

    return {
        "floor": best_e,
        "coefficient": best_coeff,
        "exponent": best_alpha,
        "predictions": predictions,
        "rmse": best_rmse,
    }
