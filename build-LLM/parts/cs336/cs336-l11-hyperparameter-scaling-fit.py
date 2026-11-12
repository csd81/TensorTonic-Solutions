import torch

def hyperparameter_scaling_fit(
    model_scales: torch.Tensor,
    learning_rates: torch.Tensor,
    batch_sizes: torch.Tensor,
    measured_losses: torch.Tensor,
    target_scale: int | float,
) -> dict:
    """
    Returns a dict of scaling law parameters, residuals, and target predictions.
    """
    device = model_scales.device
    scales = model_scales.to(dtype=torch.float64, device=device)
    lrs = learning_rates.to(dtype=torch.float64, device=device)
    batches = batch_sizes.to(dtype=torch.float64, device=device)
    losses = measured_losses.to(dtype=torch.float64, device=device)

    num_scales = scales.shape[0]
    num_batches = batches.shape[0]

    optima = []
    opt_lrs = []
    opt_batches = []

    for i in range(num_scales):
        # Flatten the (num_lrs, num_batches) grid in row-major order.
        # torch.argmin returns the first occurrence of the minimum,
        # naturally breaking ties by smaller lr_idx first, then smaller batch_idx.
        flat_losses = losses[i].reshape(-1)
        flat_idx = int(torch.argmin(flat_losses).item())

        lr_idx = flat_idx // num_batches
        batch_idx = flat_idx % num_batches

        best_lr = lrs[lr_idx]
        best_batch = batches[batch_idx]
        best_loss = losses[i, lr_idx, batch_idx]

        opt_lrs.append(best_lr)
        opt_batches.append(best_batch)

        optima.append({
            "model_scale": float(scales[i].item()),
            "learning_rate": float(best_lr.item()),
            "batch_size": float(best_batch.item()),
            "loss": float(best_loss.item()),
        })

    opt_lr_tensor = torch.stack(opt_lrs)
    opt_batch_tensor = torch.stack(opt_batches)

    # Precompute log-scale statistics for unweighted OLS
    log_s = torch.log(scales)
    x_mean = log_s.mean()
    x_centered = log_s - x_mean
    denom = torch.sum(x_centered ** 2)

    def fit_log_linear(y_tensor: torch.Tensor):
        log_y = torch.log(y_tensor)
        y_mean = log_y.mean()

        exponent = torch.sum(x_centered * (log_y - y_mean)) / denom
        log_coeff = y_mean - exponent * x_mean
        coeff = torch.exp(log_coeff)

        fitted_log_y = log_coeff + exponent * log_s
        residuals = (log_y - fitted_log_y).to(dtype=torch.float64, device=device)

        target_t = torch.tensor(float(target_scale), dtype=torch.float64, device=device)
        target_pred = coeff * torch.pow(target_t, exponent)

        return coeff, exponent, residuals, target_pred

    lr_coeff, lr_exp, lr_residuals, target_lr = fit_log_linear(opt_lr_tensor)
    batch_coeff, batch_exp, batch_residuals, target_batch = fit_log_linear(opt_batch_tensor)

    return {
        "optima": optima,
        "learning_rate_coefficient": float(lr_coeff.item()),
        "learning_rate_exponent": float(lr_exp.item()),
        "batch_size_coefficient": float(batch_coeff.item()),
        "batch_size_exponent": float(batch_exp.item()),
        "learning_rate_residuals": lr_residuals,
        "batch_size_residuals": batch_residuals,
        "target_learning_rate": float(target_lr.item()),
        "target_batch_size": float(target_batch.item()),
    }
