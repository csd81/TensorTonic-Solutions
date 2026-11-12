import torch

def mup_invariant_diagnostics(
    widths: torch.Tensor,
    activation_rms: torch.Tensor,
    activation_change_rms: torch.Tensor,
    slope_tolerance: int | float,
) -> dict:
    """
    Returns a dict of slopes, intercepts, ratios (floats) and invariant decisions (bools).
    """
    device = widths.device
    w = widths.to(dtype=torch.float64, device=device)
    act = activation_rms.to(dtype=torch.float64, device=device)
    chg = activation_change_rms.to(dtype=torch.float64, device=device)
    tol = float(slope_tolerance)

    log_w = torch.log(w)
    x_mean = log_w.mean()
    x_centered = log_w - x_mean
    denom = torch.sum(x_centered ** 2)

    def analyze_series(y: torch.Tensor):
        log_y = torch.log(y)
        y_mean = log_y.mean()

        slope_t = torch.sum(x_centered * (log_y - y_mean)) / denom
        intercept_t = y_mean - slope_t * x_mean
        ratio_t = torch.max(y) / torch.min(y)

        slope = float(slope_t.item())
        intercept = float(intercept_t.item())
        ratio = float(ratio_t.item())
        invariant = bool(abs(slope) <= tol)

        return slope, intercept, ratio, invariant

    act_slope, act_intercept, act_ratio, act_inv = analyze_series(act)
    chg_slope, chg_intercept, chg_ratio, chg_inv = analyze_series(chg)

    return {
        "activation_slope": act_slope,
        "activation_intercept": act_intercept,
        "change_slope": chg_slope,
        "change_intercept": chg_intercept,
        "activation_ratio": act_ratio,
        "change_ratio": chg_ratio,
        "activation_invariant": act_inv,
        "change_invariant": chg_inv,
    }
