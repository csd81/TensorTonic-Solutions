import math
import torch

def data_mixture_extrapolation(
    mixture_names: list[str],
    data_sizes: torch.Tensor,
    losses: torch.Tensor,
    target_size: int | float,
    floor_candidates: torch.Tensor,
) -> dict:
    """
    Returns a dict: fits (list), target_predictions (tensor), winning_mixture, winning_index.
    """
    device = data_sizes.device
    d_all = data_sizes.to(dtype=torch.float64)
    l_all = losses.to(dtype=torch.float64)
    floors = floor_candidates.to(dtype=torch.float64)
    target_t = torch.tensor(float(target_size), dtype=torch.float64, device=device)

    fits = []
    target_preds_list = []

    best_target_pred = float("inf")
    winning_index = 0

    for k, name in enumerate(mixture_names):
        d = d_all[k]
        l = l_all[k]

        log_d = torch.log(d)
        x_mean = log_d.mean()
        x_centered = log_d - x_mean
        denom = torch.sum(x_centered ** 2)

        min_loss = float(torch.min(l).item())

        best_rmse = float("inf")
        best_fit = None

        for floor_tensor in floors:
            e = float(floor_tensor.item())

            # Floor must be strictly below every observed loss in this row
            if not (e < min_loss):
                continue

            y = torch.log(l - e)
            y_mean = y.mean()

            # Unweighted OLS in natural-log space: log(L - E) = a + b * log(D)
            b = torch.sum(x_centered * (y - y_mean)) / denom
            a = y_mean - b * x_mean

            alpha = float((-b).item())
            coeff = float(torch.exp(a).item())

            # Keep only fits with positive, finite coefficient and exponent
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
                best_fit = (e, coeff, alpha, rmse)

        best_e, best_coeff, best_alpha, best_rmse = best_fit

        fits.append({
            "name": name,
            "floor": best_e,
            "coefficient": best_coeff,
            "exponent": best_alpha,
            "rmse": best_rmse,
        })

        # Extrapolate to target_size in float64 on the input device
        pred_target_tensor = best_e + best_coeff * torch.pow(target_t, -best_alpha)
        target_preds_list.append(pred_target_tensor)

        pred_target_val = float(pred_target_tensor.item())
        # Strict inequality (<) favors the earlier mixture on exact ties
        if pred_target_val < best_target_pred:
            best_target_pred = pred_target_val
            winning_index = k

    target_predictions = torch.stack(target_preds_list).to(dtype=torch.float64, device=device)

    return {
        "fits": fits,
        "target_predictions": target_predictions,
        "winning_mixture": mixture_names[winning_index],
        "winning_index": int(winning_index),
    }
